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
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.measure import marching_cubes
from Mpicker_generatemesh import check_surf, check_cc, mesh_area, plot_o3d
import open3d as o3d
import mrcfile
from pathlib import Path

import random
random.seed(3)


def read_surf(fin, sigma, dil=0):
    if fin[-3:] == 'npz':
        file = np.load(fin)
        size = file['size_zyx']
        coords = file['coords']
        data = np.zeros(size)
        if coords.ndim == 2 and len(coords) > 0 and len(coords[0]) > 0:
            data[coords.T[0], coords.T[1], coords.T[2]] = 1
    else:
        with mrcfile.open(fin, permissive=True) as mrc:
            data = mrc.data.copy()

    if dil > 0:
        data = binary_dilation(data>0, iterations=dil)
    data = data.astype(np.float32)
    if sigma > 0:
        data = gaussian_filter(data, sigma, mode='constant')
    return data   


def write_surf(fout, mesh):
    o3d.io.write_triangle_mesh(fout, mesh, write_ascii=True, write_vertex_colors=False, write_vertex_normals=False)
    return


def generate_mesh(tomo, thres, tri_area, show_3d=False):
    tomo = tomo.transpose((2,1,0))
    tomo_max = tomo.max()
    tomo_min = tomo.min()
    tomo[(0, -1), :, :] = tomo_min
    tomo[:, (0, -1), :] = tomo_min
    tomo[:, :, (0, -1)] = tomo_min
    if thres >= tomo_max or thres <= tomo_min:
        thres = (tomo_min+tomo_max)/2
        print("thres must be within volume data range, change to", thres/2)

    print("Start marching_cubes")
    verts, faces, _, _ = marching_cubes(tomo, thres, allow_degenerate=False, gradient_direction="ascent")
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
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
    print("filter_smooth_taubin")
    mesh=mesh.filter_smooth_taubin()
    check_surf(mesh)
    check_cc(mesh)
    print(mesh)

    if show_3d:
        print("press W to show mesh")
        mesh.compute_vertex_normals()
        plot_o3d([mesh])
    return mesh


def main(args):
    fin=args.fin
    fout=args.fout
    thres=args.thres
    sigma=args.sigma
    dil=args.dil
    tri_area=args.tri_area
    show_3d=args.show_3d

    tomo:np.ndarray = read_surf(fin, sigma, dil)
    mesh:o3d.geometry.TriangleMesh = generate_mesh(tomo, thres, tri_area, show_3d)
    Path(fout).parent.mkdir(parents=True, exist_ok=True)
    write_surf(fout, mesh)
    print("You can do mesh parameterization by OptCuts")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate closed surface from a segmentation", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fin', type=str, required=True,
                        help='input mrc file of the surface (or npz in mpicker)')
    parser.add_argument('--fout', type=str, required=True,
                        help='output obj file')
    parser.add_argument('--thres', type=float, default=0.5,
                        help='thres to extract surface, similar as that in Chimera.')
    parser.add_argument('--sigma', type=float, default=2,
                        help='sigma for gaussian filter, 0 to skip')
    parser.add_argument('--dil', type=int, default=0,
                        help='dilation for binary map, 0 to skip')
    parser.add_argument('--tri_area', type=float, default=20,
                        help='target mean area of triangles, 0 to skip')
    parser.add_argument('--show_3d', action='store_true',
                        help='plot result in 3d')
    args = parser.parse_args()

    main(args)