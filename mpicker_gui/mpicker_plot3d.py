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

import numpy as np
import mrcfile
from multiprocessing import Process
from scipy.ndimage import uniform_filter
import os
# import cv2

from tqdm import tqdm


def get_triangles(coord_mgrid):
    # useless now
    # here we convert to xyz, but still start from 0
    point_coord = coord_mgrid.reshape(3, -1)[::-1].T  # [[x,y,z],[x,y,z]...]
    ny, nx = coord_mgrid.shape[1:3]
    point_idx = np.arange(len(point_coord), dtype=int).reshape((ny, nx))
    triangle1_idx1 = point_idx[:-1, :-1].reshape((1, -1))[0]
    triangle1_idx2 = point_idx[1:, :-1].reshape((1, -1))[0]
    triangle1_idx3 = point_idx[:-1, 1:].reshape((1, -1))[0]
    triangle2_idx1 = point_idx[1:, 1:].reshape((1, -1))[0]
    triangle2_idx2 = point_idx[:-1, 1:].reshape((1, -1))[0]
    triangle2_idx3 = point_idx[1:, :-1].reshape((1, -1))[0]
    triangle1 = np.vstack((triangle1_idx1, triangle1_idx2, triangle1_idx3)).T
    triangle2 = np.vstack((triangle2_idx1, triangle2_idx2, triangle2_idx3)).T
    return np.vstack((triangle1, triangle2))


def get_vertices(coord_mgrid):
    # useless now
    return coord_mgrid.reshape(3, -1)[::-1].T


def clear_vertices(vert, shape_zyx):
    # useless now
    nz, ny, nx = shape_zyx
    vertices_mask = (vert[:, 0] < 0) | (vert[:, 0] > nx - 1) | (vert[:, 1] < 0) | (vert[:, 1] > ny - 1) | (vert[:, 2] < 0) | (vert[:, 2] > nz - 1)
    return vertices_mask


def clear_triangles(vert, tria, max_l=4):
    # useless now
    point1 = vert[tria[:, 0]]
    point2 = vert[tria[:, 1]]
    point3 = vert[tria[:, 2]]
    triangle_mask = (np.linalg.norm(point1 - point2, axis=1) > max_l) | (np.linalg.norm(point1 - point3, axis=1) > max_l) | (np.linalg.norm(point3 - point2, axis=1) > max_l)
    return triangle_mask


def get_help_image():
    """
    press A D S to move the slice\n
    press C to change contrast (3 states)\n
    press F to print slice number in terminal\n
    press E to turn on/off boundary box\n
    press L to turn off/on the light\n
    press X to hide/show the surface\n
    press O to save the mesh as obj file\n
    press H to see other help of open3d in terminal
    """
    help_text = ("HELP (also printed in terminal):", "calculation will take some time for thick tomo, wait please", "press A D S to move the slice", "press C to change contrast (3 states)", "press F to print slice number in terminal", "press E to turn on/off boundary box",
                "press L to turn off/on the light", "press X to hide/show the surface", "press O to save the mesh as obj file", "press H to see other help of open3d in terminal")
    # add_text1 = "hide this help (press X) before moving the slice,"
    # add_text2 = "or slice may disappear sometimes"
    # img_text = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    # for i, text in enumerate(help_text):
    #     position = (20, 20 + 20 * i)
    #     img_text = cv2.putText(img_text, text, position, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
    # position = (20, position[1] + 20)
    # img_text = cv2.putText(img_text, add_text1, position, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
    # position = (20, position[1] + 20)
    # img_text = cv2.putText(img_text, add_text2, position, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
    help_text = '\n'.join(help_text)
    # img_text = np.array(img_text)
    return help_text, None # np.array(img_text)


def plot_o3d_real(coords, colors, sizes, ball_resolution=5, point_size=1.5, ball_norm=True, pcd_norm=True):
    # from time import time
    # name="CoordColorSize_"+str(time())+".npz"
    # np.savez(name,Coord=coords,Color=colors,Size=sizes)
    try:
        import open3d as o3d
    except:
        return
    coord_ball = coords[sizes > 0]
    color_ball = colors[sizes > 0]
    size_ball = sizes[sizes > 0]
    # o3d_list = []
    pcds = o3d.geometry.PointCloud()
    if len(coord_ball) > 0:
        for i in range(len(coord_ball)):
            z, y, x = coord_ball[i]
            color = color_ball[i]
            size = size_ball[i]
            sphere = o3d.geometry.TriangleMesh.create_sphere(size, resolution=ball_resolution)
            # if ball_norm:
            #     sphere.compute_vertex_normals()
            sphere.translate([x, y, z])
            # sphere.paint_uniform_color(color)
            # o3d_list.append(sphere)
            sphere = sphere.sample_points_uniformly(number_of_points=100)
            if ball_norm:
                sphere.estimate_normals()
            sphere.paint_uniform_color(color)
            pcds+=sphere           
    pcd_list = [ sizes==s for s in set(sizes) if s<=0 ]
    if len(pcd_list) > 0:
        for mask in pcd_list:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[mask][:, ::-1])  # to xyz
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            if pcd_norm:
                pcd.estimate_normals()
            #o3d_list.append(pcd)
            pcds+=pcd

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=600)
    # for a_o3d in o3d_list:
    #     vis.add_geometry(a_o3d)
    vis.add_geometry(pcds)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-90)  # change to orthogonal projection
    rend = vis.get_render_option()
    rend.mesh_show_back_face = True
    rend.point_size = point_size
    vis.run()
    vis.destroy_window()
    return


def plot_o3d(coord, color, size, join=False):
    proc = Process(target=plot_o3d_real, args=(coord, color, size))
    proc.start()
    if join:
        proc.join()
    return


def organize_coords(list_coord, list_color, list_size=None):
    if list_size is None:
        list_size = [0 for i in list_color]
    coord = np.vstack(list_coord)
    color = []
    size = []
    for i in range(len(list_coord)):
        color += [list_color[i]] * len(list_coord[i])
        size += [list_size[i]] * len(list_coord[i])
    color = np.array(color)
    size = np.array(size)
    return coord, color, size


def show_3d_texture(output, convert_file, shape_zyx, points_zyx=None, points_radius=0, \
            clear_outside=True, down=4):
    try:
        import open3d as o3d
    except:
        return
    
    def get_tri(shapeyx, down):
        """star from 0"""
        ny, nx = shapeyx
        nu = len(np.arange(0,nx,down))
        nv = len(np.arange(0,ny,down))
        point_idx = np.arange(nu*nv, dtype=int).reshape((nu, nv))
        triangle1_idx1 = point_idx[:-1, :-1].reshape((1, -1))[0]
        triangle1_idx2 = point_idx[1:, :-1].reshape((1, -1))[0]
        triangle1_idx3 = point_idx[:-1, 1:].reshape((1, -1))[0]
        triangle2_idx1 = point_idx[1:, 1:].reshape((1, -1))[0]
        triangle2_idx2 = point_idx[:-1, 1:].reshape((1, -1))[0]
        triangle2_idx3 = point_idx[1:, :-1].reshape((1, -1))[0]
        triangle1 = np.vstack((triangle1_idx1, triangle1_idx2, triangle1_idx3)).T
        triangle2 = np.vstack((triangle2_idx1, triangle2_idx2, triangle2_idx3)).T
        return np.vstack((triangle1, triangle2))

    def get_meshes(coord_mgrid, down=5, clear_outside=True, shaperaw_zyx=None):
        """here xyz start from 0"""
        ny, nx = coord_mgrid.shape[2:4]
        tri = get_tri((ny, nx), down)
        u, v = np.mgrid[0:nx:down, 0:ny:down]
        u = u.flatten().astype(int)
        v = v.flatten().astype(int)
        uv = np.column_stack((u/nx, v/ny))
        mesh_list = []
        for i in tqdm(range(coord_mgrid.shape[1])):
            z, y, x = coord_mgrid[:,i,:,:][:, v, u]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.column_stack((x,y,z)))
            mesh.triangles = o3d.utility.Vector3iVector(tri)
            if clear_outside and shaperaw_zyx is not None:
                nrz, nry, nrx = shaperaw_zyx
                vertices_mask = (x < 0) | (x > nrx - 1) | (y < 0) | (y > nry - 1) | (z < 0) | (z > nrz - 1)
                mesh.remove_vertices_by_mask(vertices_mask)
                uv_clear = uv[~vertices_mask]
                tri_clear = np.array(mesh.triangles)
            else:
                uv_clear = uv
                tri_clear = tri
            try:
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_clear[tri_clear.flatten()])
                mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(tri_clear), dtype=int))
            except:
                mesh.triangle_uvs = list(uv_clear[tri_clear.flatten()])
            mesh.compute_vertex_normals()
            mesh_list.append(mesh)
        return mesh_list
    
    def add_texture(mesh, image, d_range):
        d_small, d_big = float(d_range[0]), float(d_range[1])
        image = (image-d_small)/(d_big-d_small)
        image = np.clip(image, 0, 1) * 255
        image = image.astype(np.uint8)
        image = np.stack([image,image,image], axis=2)
        texture = o3d.geometry.Image(image)
        try:
            mesh.textures = [texture]
        except:
            mesh.texture = texture
        return
    
    surf_mgrid = np.load(convert_file)
    with mrcfile.open(output) as mrc:
        density_all = mrc.data
    down = int(down)
    if down > 1:
        print(f"bin{down} to speedup")
    z_idx = int(density_all.shape[0] / 2) + 1  # the middle one, start from 1

    mesh_all = get_meshes(surf_mgrid, down, clear_outside, shape_zyx)
    d_min, d_max = density_all.min(), density_all.max()
    d_small1, d_big1 = np.percentile(density_all, [0.5,99.5])
    d_small2, d_big2 = np.percentile(density_all, [5,95])
    contrast_text=["full","0.5%","5%"]
    print("real min and max", d_min, d_max)
    for mesh, image in zip(mesh_all, density_all):
        add_texture(mesh, image, (d_min, d_max))
    mesh = mesh_all[z_idx - 1]
    print(mesh)
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), np.array(shape_zyx)[::-1] - 1)
    bbox.color = (0, 0, 0)
    help_text, __ = get_help_image()

    show_3d_texture.z_idx = z_idx 
    show_3d_texture.show_box = False
    show_3d_texture.mesh = mesh
    show_3d_texture.show_surface = True
    show_3d_texture.contrast = 0 # 0 1 2

    def my_change_up(vis):
        if show_3d_texture.z_idx < len(mesh_all) and show_3d_texture.show_surface:
            vis.remove_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            show_3d_texture.z_idx += 1
            show_3d_texture.mesh = mesh_all[show_3d_texture.z_idx - 1]
            vis.add_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            return False
        else:
            return False

    def my_change_down(vis):
        if show_3d_texture.z_idx > 1 and show_3d_texture.show_surface:
            vis.remove_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            show_3d_texture.z_idx -= 1
            show_3d_texture.mesh = mesh_all[show_3d_texture.z_idx - 1]
            vis.add_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            return False
        else:
            return False

    def my_change_middle(vis):
        if show_3d_texture.z_idx != int(len(mesh_all) / 2) + 1 and show_3d_texture.show_surface:
            vis.remove_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            show_3d_texture.z_idx = int(len(mesh_all) / 2) + 1
            show_3d_texture.mesh = mesh_all[show_3d_texture.z_idx - 1]
            vis.add_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            return False
        else:
            return False

    def print_z(vis):
        print(show_3d_texture.z_idx)
        return False

    def draw_bbox(vis):
        if show_3d_texture.show_box:
            vis.remove_geometry(bbox, reset_bounding_box=False)
            show_3d_texture.show_box = False
            return False
        else:
            vis.add_geometry(bbox, reset_bounding_box=False)
            show_3d_texture.show_box = True
        return False
    
    def draw_surface(vis):
        if show_3d_texture.show_surface:
            vis.remove_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            show_3d_texture.show_surface = False
        else:
            vis.add_geometry(show_3d_texture.mesh, reset_bounding_box=False)
            show_3d_texture.show_surface = True
        return False
    
    def change_contrast(vis):
        if show_3d_texture.show_surface:
            show_3d_texture.contrast = (show_3d_texture.contrast + 1)%3
            d_small = [d_min,d_small1,d_small2][show_3d_texture.contrast]
            d_big = [d_max,d_big1,d_big2][show_3d_texture.contrast]
            for mesh, image in zip(mesh_all, density_all):
                add_texture(mesh, image, (d_small, d_big))
            print("contrast:",contrast_text[show_3d_texture.contrast],"range: %.2f %.2f"%(d_small,d_big))
            vis.update_geometry(show_3d_texture.mesh)

    def save_obj(vis):
        if show_3d_texture.show_surface:
            pre = "show3d"
            idx = 0
            while os.path.exists(f"{pre}_{idx:03}.obj"):
                idx += 1
            fname = os.path.abspath(f"{pre}_{idx:03}.obj")
            o3d.io.write_triangle_mesh(fname, show_3d_texture.mesh)
            print("save obj (also mtl and png) as:", fname)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord("A"), my_change_down)
    vis.register_key_callback(ord("D"), my_change_up)
    vis.register_key_callback(ord("S"), my_change_middle)
    vis.register_key_callback(ord("C"), change_contrast)
    vis.register_key_callback(ord("F"), print_z)
    vis.register_key_callback(ord("E"), draw_bbox)
    vis.register_key_callback(ord("X"), draw_surface)
    vis.register_key_callback(ord("O"), save_obj)
    vis.create_window(width=800, height=800)
    vis.add_geometry(show_3d_texture.mesh)

    if points_zyx is not None and points_radius>0:
        for z, y, x in points_zyx:
            color = (0, 0, 1)
            size = points_radius
            sphere = o3d.geometry.TriangleMesh.create_sphere(size)
            sphere.compute_vertex_normals()
            sphere.translate([x, y, z])
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere, reset_bounding_box=False)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-90)  # change to orthogonal projection
    rend = vis.get_render_option()
    rend.mesh_show_back_face = True
    print(help_text)
    vis.run()
    vis.destroy_window()
    return


def show_3d(output, convert_file, shape_zyx, points_zyx=None, points_radius=0, \
            clear_outside=True, clear_big=0, down=2):
    # useless now
    try:
        import open3d as o3d
    except:
        return
    surf_mgrid = np.load(convert_file) #[:,:,149:800,:] ##change
    with mrcfile.open(output) as mrc:
        density_all = mrc.data #[:,149:800,:]
    down = int(down)
    if down > 1:
        print(f"bin{down} to speedup")
        surf_mgrid = surf_mgrid[:, :, down-1::down, down-1::down]
        density_all = density_all.astype(np.float32)
        density_all = [uniform_filter(s, down, mode='constant')[down-1::down, down-1::down] for s in density_all]
        density_all = np.stack(density_all)
    z_idx = int(density_all.shape[0] / 2) + 1  # the middle one, start from 1
    tomo_min, tomo_max = density_all.min().astype(float), density_all.max().astype(float)
    print("real min and max", tomo_min, tomo_max)
    density_all = (density_all - tomo_min) / (tomo_max - tomo_min)

    triangles = get_triangles(surf_mgrid[:, z_idx, :, :]) # all slices are same
    mesh0 = o3d.geometry.TriangleMesh()
    mesh0.vertices = o3d.utility.Vector3dVector(get_vertices(surf_mgrid[:, z_idx, :, :]))
    mesh0.triangles = o3d.utility.Vector3iVector(triangles)

    if clear_big > 0:
        coord_mgrid = surf_mgrid[:, z_idx - 1, :, :]
        point_coord = get_vertices(coord_mgrid)
        triangle_mask = clear_triangles(point_coord, triangles, max_l=clear_big)

    def get_mesh(z, clear_outside, clear_big):
        coord_mgrid = surf_mgrid[:, z - 1, :, :]
        point_coord = get_vertices(coord_mgrid)
        density = density_all[z - 1, :, :].reshape(1, -1)[0]
        color = np.array([density, density, density]).T
        mesh = o3d.geometry.TriangleMesh(mesh0)
        mesh.vertices = o3d.utility.Vector3dVector(point_coord)
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
        if clear_big > 0:  # delete same triangles as middle one
            mesh.remove_triangles_by_mask(triangle_mask)
        if clear_outside:
            vertices_mask = clear_vertices(point_coord, shape_zyx)
            mesh.remove_vertices_by_mask(vertices_mask)
        if clear_big > 0:
            mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    mesh_all = [get_mesh(i + 1, clear_outside, clear_big) for i in tqdm(range(density_all.shape[0]))]
    density_all=[np.array(mesh.vertex_colors)[:,0] for mesh in mesh_all]
    d_min, d_max = np.hstack(density_all).min(), np.hstack(density_all).max()
    d_small1, d_big1 = np.percentile(np.hstack(density_all),[0.5,99.5])
    d_small2, d_big2 = np.percentile(np.hstack(density_all),[5,95])
    contrast_text=["full","0.5%","5%"]
    #print("min to max", d_min, d_small1, d_small2, d_big2, d_big1, d_max)
    #print(np.percentile(np.hstack(density_all),[0.01,0.1,0.5,1,5,95,99,99.5,99.9,99.99]))
    
    mesh = mesh_all[z_idx - 1]
    print(mesh)
    shape_xyz=np.array(shape_zyx)[::-1].copy()
    # shape_xyz[1]=560 ##change
    # shape_xyz[0]=760
    # shape_xyz[2]=404
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), shape_xyz - 1)
    # bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([130, 150, 170]), np.array([870, 410, 280]))
    bbox.color = (0, 0, 0)

    help_text, __ = get_help_image()
    # image_text = o3d.geometry.Image(image_text)

    show_3d.z_idx = z_idx  # ??
    show_3d.show_box = False
    show_3d.mesh = mesh
    # show_3d.show_help = True
    show_3d.show_surface = True
    show_3d.contrast = 0 # 0 1 2

    def my_change_up(vis):
        if show_3d.z_idx < len(mesh_all) and show_3d.show_surface:
            vis.remove_geometry(show_3d.mesh, reset_bounding_box=False)
            show_3d.z_idx += 1
            show_3d.mesh = mesh_all[show_3d.z_idx - 1]
            vis.add_geometry(show_3d.mesh, reset_bounding_box=False)
            return False
        else:
            return False

    def my_change_down(vis):
        if show_3d.z_idx > 1 and show_3d.show_surface:
            vis.remove_geometry(show_3d.mesh, reset_bounding_box=False)
            show_3d.z_idx -= 1
            show_3d.mesh = mesh_all[show_3d.z_idx - 1]
            vis.add_geometry(show_3d.mesh, reset_bounding_box=False)
            return False
        else:
            return False

    def my_change_middle(vis):
        if show_3d.z_idx != int(len(mesh_all) / 2) + 1 and show_3d.show_surface:
            vis.remove_geometry(show_3d.mesh, reset_bounding_box=False)
            show_3d.z_idx = int(len(mesh_all) / 2) + 1
            show_3d.mesh = mesh_all[show_3d.z_idx - 1]
            vis.add_geometry(show_3d.mesh, reset_bounding_box=False)
            return False
        else:
            return False

    def print_z(vis):
        print(show_3d.z_idx)
        return False

    def draw_bbox(vis):
        if show_3d.show_box:
            vis.remove_geometry(bbox, reset_bounding_box=False)
            show_3d.show_box = False
            return False
        else:
            vis.add_geometry(bbox, reset_bounding_box=False)
            show_3d.show_box = True
        return False

    # def draw_help(vis):
    #     if show_3d.show_help:
    #         vis.remove_geometry(image_text, reset_bounding_box=False)
    #         show_3d.show_help = False
    #     else:
    #         vis.remove_geometry(show_3d.mesh, reset_bounding_box=False)
    #         if show_3d.show_box:
    #             vis.remove_geometry(bbox, reset_bounding_box=False)
    #         vis.add_geometry(image_text, reset_bounding_box=False)
    #         vis.update_geometry(image_text)
    #         if show_3d.show_box:
    #             vis.add_geometry(bbox, reset_bounding_box=False)
    #         vis.add_geometry(show_3d.mesh, reset_bounding_box=False)
    #         show_3d.show_help = True
    #     return False

    def draw_surface(vis):
        if show_3d.show_surface:
            vis.remove_geometry(show_3d.mesh, reset_bounding_box=False)
            show_3d.show_surface = False
        else:
            vis.add_geometry(show_3d.mesh, reset_bounding_box=False)
            show_3d.show_surface = True
        return False

    def change_contrast(vis):
        if show_3d.show_surface:
            show_3d.contrast = (show_3d.contrast + 1)%3
            d_small,d_big=[d_min,d_small1,d_small2][show_3d.contrast],[d_max,d_big1,d_big2][show_3d.contrast]
            for i,mesh in enumerate(mesh_all):
                density=density_all[i]
                density=(density-d_small)/(d_big-d_small)
                color = np.array([density, density, density]).T
                mesh.vertex_colors = o3d.utility.Vector3dVector(color)
            print("contrast:",contrast_text[show_3d.contrast],"range: %.2f %.2f"%(d_small,d_big))
            vis.update_geometry(show_3d.mesh)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord("A"), my_change_down)
    vis.register_key_callback(ord("D"), my_change_up)
    vis.register_key_callback(ord("S"), my_change_middle)
    vis.register_key_callback(ord("C"), change_contrast)
    vis.register_key_callback(ord("F"), print_z)
    vis.register_key_callback(ord("E"), draw_bbox)
    vis.register_key_callback(ord("X"), draw_surface)
    vis.create_window(width=800, height=800)
    # vis.add_geometry(image_text)
    vis.add_geometry(show_3d.mesh)

    if points_zyx is not None and points_radius>0:
        for z, y, x in points_zyx:
            color = (0, 0, 1)
            size = points_radius
            sphere = o3d.geometry.TriangleMesh.create_sphere(size)
            sphere.compute_vertex_normals()
            sphere.translate([x, y, z])
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere, reset_bounding_box=False)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-90)  # change to orthogonal projection
    rend = vis.get_render_option()
    rend.mesh_show_back_face = True
    print(help_text)
    vis.run()
    vis.destroy_window()

    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return