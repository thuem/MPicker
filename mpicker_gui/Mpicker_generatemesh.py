#!/usr/bin/env python3

# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial import KDTree, Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order, connected_components
import open3d as o3d
import mrcfile
from pathlib import Path

import random
random.seed(3)


def read_surf(fin):
    if fin[-3:] == 'txt':
        xyz = np.loadtxt(fin, ndmin=2)
    elif fin[-3:] == 'npz':
        coords = np.load(fin)['coords'] + 1
        xyz = coords[:, ::-1]
    elif fin[-3:] == 'mrc':
        with mrcfile.open(fin, permissive=True) as mrc:
            data = mrc.data
        xyz=np.argwhere(data)[:, ::-1] + 1
    else:
        raise Exception("fin should be npz file or txt file or mrc file")
    return xyz.astype(float)


def write_surf(fout, mesh):
    o3d.io.write_triangle_mesh(fout, mesh, write_ascii=True, write_vertex_colors=False, write_vertex_normals=False)
    return


def check_surf(mesh):
    nan_idx = np.unique(np.argwhere(np.isnan(np.array(mesh.vertices)))[:,0]) # I don't know why
    if len(nan_idx) > 0:
        mesh.remove_vertices_by_index(nan_idx)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return 


def check_cc(mesh):
    # connected_components of open3d 0.9.0 is unreliable
    mesh.compute_adjacency_list()
    indices = []
    indptr = [0]
    for adj in mesh.adjacency_list:
        indices += list(adj)
        indptr.append(indptr[-1]+len(adj))
    data = [1]*len(indices)
    length = len(mesh.adjacency_list)
    adjacency = csr_matrix((data, indices, indptr), shape=(length, length))
    n, labels = connected_components(adjacency, directed=False)
    if n > 1:
        print(f"has {n} connected components, so leave the largest one.")
        labelid, vnum = np.unique(labels, return_counts=True)
        id_save = labelid[np.argmax(vnum)]
        mesh.remove_vertices_by_mask(labels!=id_save)


def sample_coord_3d(coords, dist=10, filt=True, filt_dist=2):
    if dist <= filt_dist:
        filt = False
    total_coords = coords.shape[0]
    pick_coords_index = list(range(total_coords))
    pick_coords = coords[pick_coords_index].astype(float)

    tree = KDTree(pick_coords)
    pick_idx = np.ones(len(pick_coords), dtype=bool)
    for i in range(len(pick_coords)):
        if pick_idx[i]:
            pick_idx[tree.query_ball_point(pick_coords[i], dist)] = False
            pick_idx[i] = True # dist from itself is always 0
    pick_coords_sparse=pick_coords[pick_idx]
    print('total coords num', total_coords, 'final pick', len(pick_coords_sparse))

    if filt:
        for i, point in enumerate(pick_coords_sparse):
            near_idx = tree.query_ball_point(point, filt_dist)
            near_coords = pick_coords[near_idx]
            pick_coords_sparse[i] = near_coords.mean(axis=0)

    return pick_coords_sparse #, np.arange(total_coords)[pick_coords_index][pick_idx]


def orient_normals(pcd, knn):
    try:
        pcd.orient_normals_consistent_tangent_plane(knn)
        return
    except:
        # if open3d 0.9
        pass
    points = np.array(pcd.points)
    norms = np.array(pcd.normals)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    num = len(points)
    dela = Delaunay(points)
    indptr, indices = dela.vertex_neighbor_vertices
    coo = csr_matrix(([1]*len(indices), indices, indptr), (num, num)).tocoo()
    row, col = coo.row, coo.col
    dist = np.linalg.norm(points[row]-points[col], axis=1)
    dist_graph = coo_matrix((dist, (row, col)), (num, num))
    emst = minimum_spanning_tree(dist_graph)

    tree = KDTree(points)
    _, nbh = tree.query(points, knn+1)
    indptr = np.arange(0, num*knn+1, knn)
    indices = nbh[:, 1:].flatten()
    riem_graph = (emst + csr_matrix(([1]*len(indices), indices, indptr), (num, num))).tocoo()
    row, col = riem_graph.row, riem_graph.col
    cost = 1.1 - np.abs(np.sum(norms[row]*norms[col], axis=1)) # 1-|n1.n2|
    riem_graph = coo_matrix((cost, (row, col)), (num, num))
    riem_tree = minimum_spanning_tree(riem_graph)
    idx_init = np.argmin(points[:, 0])
    if norms[idx_init, 0] > 0:
        norms[idx_init] *= -1 # make xmin points to -x, so for sphere, norm vector can point outside
    riem_order, predecessors = depth_first_order(riem_tree, idx_init, directed=False)
    for i in range(1, len(riem_order)):
        # riem_order[0] is idx_init
        idx_this = riem_order[i]
        idx_last = predecessors[idx_this]
        if np.dot(norms[idx_last], norms[idx_this]) < 0:
            norms[idx_this] *= -1
    pcd.normals = o3d.utility.Vector3dVector(norms)
    return


def mesh_area(mesh):
    try:
        area = mesh.get_surface_area()
        return area
    except:
        # if open3d 0.9
        pass
    point1, point2, point3 = np.array(mesh.triangles).T
    data = np.array(mesh.vertices)
    a = np.linalg.norm(data[point1]-data[point2], axis=1)
    b = np.linalg.norm(data[point2]-data[point3], axis=1)
    c = np.linalg.norm(data[point3]-data[point1], axis=1)
    p = (a + b + c) / 2
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    return area.sum()


def plot_o3d(geometries, point_size=2):
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=600, height=600)
        for geometry in geometries:
            vis.add_geometry(geometry)
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=-90)  # change to orthogonal projection
        rend = vis.get_render_option()
        rend.mesh_show_back_face = True
        rend.point_size = point_size
        vis.run()
        vis.destroy_window()
    except:
        print("fail to plot 3d.")


def remove_statistical_outlier(pcd, filt_knn, filt_std, show_3d=False):
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=filt_knn, std_ratio=filt_std)
    try:
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)
    except:
        # if open3d 0.9
        inlier_cloud = pcd.select_down_sample(ind)
        outlier_cloud = pcd.select_down_sample(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    if show_3d:
        plot_o3d([inlier_cloud, outlier_cloud])
    pcd = inlier_cloud
    return pcd


def generate_pcd(xyz, down, knn, filt_knn, filt_std, show_3d=False):
    pcd = o3d.geometry.PointCloud()
    if down > 0:
        print("Downsample the point cloud")
        xyz = sample_coord_3d(xyz, dist=down)
        # pcd = pcd.voxel_down_sample(voxel_size=down)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn))
    if filt_knn > 0:
        pcd = remove_statistical_outlier(pcd, filt_knn, filt_std, show_3d)
    else:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if show_3d:
            plot_o3d([pcd])
    return pcd


def generate_mesh(pcd, knn, thres, tri_area, show_3d=False):
    print("Poisson surface reconstruction")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn))
    orient_normals(pcd, knn)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.compute_vertex_normals()
    densities = np.asarray(densities)
    den_min, den_max, den_mean, den_std = densities.min(), densities.max(), densities.mean(), densities.std()
    print("thres %.2f, mean %.2f, std %.2f, min %.2f, max %.2f"%(thres, den_mean, den_std, den_min, den_max))

    color_min, color_max = den_mean - den_std, den_mean + den_std
    colors = (densities - color_min) / (color_max - color_min)
    colors = np.clip(colors, 0, 1)
    colors = np.array([np.zeros_like(colors), colors, np.zeros_like(colors)]).T
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if thres > 0:
        print("Remove mesh by thres")
        mesh.remove_vertices_by_mask(densities<thres)
    check_surf(mesh)
    check_cc(mesh)
    print(mesh)
    area = mesh_area(mesh)
    print("total area %.1e" % area)
    if tri_area > 0:
        print("Simplify mesh by area")
        target_num = int(area/tri_area)
        if len(np.asarray(mesh.triangles)) <= 2*target_num:
            mesh = mesh.subdivide_loop(1)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_num)
        mesh = mesh.filter_smooth_taubin()
        check_surf(mesh)
        check_cc(mesh)
        print(mesh)
    # mesh=mesh.filter_smooth_taubin()
    # mesh=mesh.filter_smooth_laplacian()

    if show_3d:
        print("press W to show mesh")
        pcd.paint_uniform_color([0, 0, 0.5])
        mesh.compute_vertex_normals()
        plot_o3d([pcd, mesh])
    return mesh


def main(args):
    fin=args.fin
    fout=args.fout
    down=args.down
    knn=args.knn
    thres=args.thres
    filt_knn=args.filt_knn
    filt_std=args.filt_std
    tri_area=args.tri_area
    show_3d=args.show_3d

    xyz:np.ndarray = read_surf(fin)
    pcd:o3d.geometry.PointCloud = generate_pcd(xyz, down, knn, filt_knn, filt_std, show_3d)
    mesh:o3d.geometry.TriangleMesh = generate_mesh(pcd, knn, thres, tri_area, show_3d)
    Path(fout).parent.mkdir(parents=True, exist_ok=True)
    write_surf(fout, mesh)
    print("You can do mesh parameterization by OptCuts")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recommend Open3d 0.11 or higher", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fin', type=str, required=True,
                        help='input npz or txt or mrc file of the surface in mpicker')
    parser.add_argument('--fout', type=str, required=True,
                        help='output obj file')
    parser.add_argument('--down', type=float, default=6,
                        help='distance in pixel, for down sample at first, 0 to skip')
    parser.add_argument('--knn', type=int, default=15,
                        help='number of neighbours to get norm vector')
    parser.add_argument('--thres', type=float, default=5.5,
                        help='thres to cut after surface reconstruction, bigger to remove more mesh, 0 to skip')
    parser.add_argument('--filt_knn', type=int, default=30,
                        help='knn for points remove_statistical_outlier, 0 to skip filt')
    parser.add_argument('--filt_std', type=float, default=2,
                        help='std for points remove_statistical_outlier, smaller to remove more points')
    parser.add_argument('--tri_area', type=float, default=60,
                        help='target mean area of triangles, 0 to skip')
    parser.add_argument('--show_3d', action='store_true',
                        help='plot some results, close the window to continue')
    args = parser.parse_args()

    main(args)