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

import argparse, configparser, os
import numpy as np
import scipy.ndimage as nd

import mrcfile
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
from multiprocessing import Process, set_start_method
try:
    import open3d as o3d
except Exception as e:
    print(e)
    print("fail to import open3d, will not use it")

import random
random.seed(3)


def read_obj_xyz_uv_triangle(obj_path):
    if len(obj_path) < 4 or obj_path[-4:] != ".obj":
        raise Exception("only accept .obj file")
    with open(obj_path, "r") as file:
        lines = file.readlines()
    vertices = []
    texcoords = []
    triangles = []
    tex2ver = []
    for line in lines:
        words = line.strip().split()
        if len(words) == 0:
            continue
        if words[0] == "v":
            vertex = [float(words[1]), float(words[2]), float(words[3])]
            vertices.append(vertex)
        if words[0] == "vt":
            texcoord = [float(words[1]), float(words[2])]
            texcoords.append(texcoord)
        elif words[0] == "f":
            if len(words) != 4:
                raise Exception("only accept triangle mesh")
            face = []
            for word in words[1:]:
                indices = word.split("/")
                vertex_index = int(indices[0]) - 1
                texcoord_index = int(indices[1]) - 1
                face.append(texcoord_index) # for drawing triangles in UV plane
                if texcoord_index != vertex_index:
                    tex2ver.append([texcoord_index, vertex_index]) # note they are same in 3D
            triangles.append(face)

    if len(texcoords) == 0:
        raise Exception("The input mesh has no texture coordinates")
    texcoords = np.array(texcoords, dtype=float)
    vertices = np.array(vertices, dtype=float)
    vertices_full = np.zeros((len(texcoords), 3), dtype=float)
    vertices_full[:len(vertices)] = vertices
    triangles = np.array(triangles, dtype=int)
    for tex_id, ver_id in tex2ver:
        vertices_full[tex_id] = vertices_full[ver_id]

    return vertices_full, texcoords, triangles


def get_mesh_norm_uv(fobj, angle, umin, vmin, uvscale):
    # generate a tmp obj file and load it. useless now
    with open(fobj, "r") as file:
        lines = file.readlines()
    rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for i, line in enumerate(lines):
        words = line.strip().split()
        if len(words) == 0:
            continue
        if words[0] == "vt":
            uv = np.array([float(words[1]), float(words[2])])
            u, v = np.dot(rot_matrix, uv)
            u = (u-umin)/uvscale
            v = (v-vmin)/uvscale
            line_new = f"vt {u} {v}\n"
            lines[i] = line_new
    fobj_tmp = fobj + "_tmp.obj"
    with open(fobj_tmp, "w") as file:
        file.writelines(lines)
    mesh = o3d.io.read_triangle_mesh(fobj_tmp)
    os.remove(fobj_tmp)
    print("remove", fobj_tmp)
    return mesh


def triangle2d_mask(vertex_uvs, triangles, vu_mgrid):
    # return mask of triangles
    interval = vu_mgrid[0, 1, 0] - vu_mgrid[0, 0, 0]
    vmin, umin = vu_mgrid[:, 0, 0]
    _, sv, su = vu_mgrid.shape
    result = np.zeros((sv, su), dtype=bool)
    for triangle in triangles:
        vertex_vu = vertex_uvs[triangle, ::-1]
        vvmin, uumin = vertex_vu.min(axis=0)
        vvmax, uumax = vertex_vu.max(axis=0)
        i0 = max(int((vvmin-vmin)/interval)-1, 0)
        i1 = min(int((vvmax-vmin)/interval)+2, sv-1)
        j0 = max(int((uumin-umin)/interval)-1, 0)
        j1 = min(int((uumax-umin)/interval)+2, su-1)
        mgrid = vu_mgrid[:, i0:i1+1, j0:j1+1]
        # triangle2d = Delaunay(vertex_uv)
        # img[i0:i1+1,j0:j1+1][triangle2d.find_simplex(mgrid.transpose(1,2,0))>=0]=True
        v0 = mgrid - vertex_vu[0][:, np.newaxis, np.newaxis]
        v1 = mgrid - vertex_vu[1][:, np.newaxis, np.newaxis]
        v2 = mgrid - vertex_vu[2][:, np.newaxis, np.newaxis]
        c0 = np.sign(np.cross(v0, v1, axis=0))
        c1 = np.sign(np.cross(v1, v2, axis=0))
        c2 = np.sign(np.cross(v2, v0, axis=0))
        mask = (c0*c1!=-1) & (c1*c2!=-1) & (c2*c0!=-1) # c0 c1 c2 should be same or 0 if in triangle
        result[i0:i1+1, j0:j1+1][mask] = True
    return result


def read_obj(obj_path, do_rotate):
    # mesh_uv = o3d.io.read_triangle_mesh(obj_path)
    # if not mesh_uv.has_triangle_uvs():
    #     raise Exception("The input mesh has no texture coordinates.")
    # trangles = np.array(mesh_uv.triangles)
    # triangle_uvs = np.array(mesh_uv.triangle_uvs)
    # zyx = np.array(mesh_uv.vertices)[:, ::-1]
    # vertex_uvs = np.zeros((trangles.max()+1, 2))
    # vertex_uvs[trangles.flatten()] = triangle_uvs
    # zyxuv = np.hstack([zyx,vertex_uvs])
    xyz, uv, _ = read_obj_xyz_uv_triangle(obj_path)
    zyxuv = np.hstack([xyz[:, ::-1], uv])
    angle=0
    if do_rotate:
        vertex_uvs = zyxuv[:, 3:5]
        angles = np.linspace(-1 / 2 * np.pi, 1 / 2 * np.pi, 100)
        u_len = []
        for ang in angles:
            rot_matrix = np.array([[np.cos(ang), -1 * np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            coord_u_convert = np.dot(rot_matrix, vertex_uvs.T)[0]
            u_len.append(coord_u_convert.max() - coord_u_convert.min())
        u_len = np.array(u_len)
        ang_idx = u_len.argmin()
        angle = angles[ang_idx]
        rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        zyxuv[:, 3:5] = np.dot(rot_matrix, vertex_uvs.T).T
        print("rotate:", angle/np.pi*180)
    return zyxuv, angle


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


def plot_mesh_real(coords, colors, sizes, fobj=None, obj_angle=0,
                   ball_resolution=5, point_size=1.5, ball_norm=True, pcd_norm=True, mesh_texture=True):
    coord_ball = coords[sizes > 0]
    color_ball = colors[sizes > 0]
    size_ball = sizes[sizes > 0]
    points = o3d.geometry.PointCloud()
    spheres = o3d.geometry.TriangleMesh()
    if len(coord_ball) > 0:
        for i in range(len(coord_ball)):
            z, y, x = coord_ball[i]
            color = color_ball[i]
            size = size_ball[i]
            sphere = o3d.geometry.TriangleMesh.create_sphere(size, resolution=ball_resolution)
            sphere.translate([x, y, z])
            sphere.paint_uniform_color(color)
            spheres += sphere
        if ball_norm:
            spheres.compute_vertex_normals()         
    pcd_list = [ sizes==s for s in set(sizes) if s<=0 ]
    if len(pcd_list) > 0:
        for mask in pcd_list:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords[mask][:, ::-1])  # to xyz
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            if pcd_norm:
                pcd.estimate_normals()
            points+=pcd
    if fobj is not None:
        mesh_uv = o3d.io.read_triangle_mesh(fobj)
        if mesh_texture:
            uvs = np.array(mesh_uv.triangle_uvs)
            rot_matrix = np.array([[np.cos(obj_angle), -1 * np.sin(obj_angle)], [np.sin(obj_angle), np.cos(obj_angle)]])
            uvs = np.dot(rot_matrix, uvs.T).T
            umin, vmin = uvs.min(axis=0)
            umax, vmax = uvs.max(axis=0)
            uv_scale = max((umax-umin), (vmax-vmin))
            uvs[:,0] = (uvs[:,0]-umin)/uv_scale
            uvs[:,1] = (uvs[:,1]-vmin)/uv_scale
            y,_ = np.mgrid[0:400,0:400]/400 # 0 to 1
            image = np.ones((400,400,3))
            for i in range(20):
                image[i*20:i*20+10,:,:] *= -1
                image[:,i*20:i*20+10,:] *= -1
            image = (image+1)/2 # 0 or 1
            image[:,:,0][y>0.5] *= (2-2*y)[y>0.5]
            image[:,:,2][y<0.5] *= (2*y)[y<0.5]
            image[:,:,1] *= (1-2*np.abs(y-0.5))
            image *= 0.6
            texture = o3d.geometry.Image(image.astype(np.float32))
            try:
                mesh_uv.triangle_uvs = o3d.utility.Vector2dVector(uvs)
                mesh_uv.textures = [texture]
            except:
                # open3d 0.9
                # mesh_uv = get_mesh_norm_uv(fobj, obj_angle, umin, vmin, uv_scale)
                mesh_uv.triangle_uvs = list(uvs)
                mesh_uv.texture = texture
                print("Recommend Open3d 0.11 or higher to draw texture")
        mesh_uv.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=600)
    if len(coord_ball) > 0:
        vis.add_geometry(spheres)
    if len(pcd_list) > 0:
         vis.add_geometry(points)
    if fobj is not None:
        vis.add_geometry(mesh_uv)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-90)  # change to orthogonal projection
    rend = vis.get_render_option()
    rend.mesh_show_back_face = True
    rend.point_size = point_size
    vis.run()
    vis.destroy_window()
    return


def plot_mesh(coord, color, size, fobj=None, obj_angle=0, join=False):
    proc = Process(target=plot_mesh_real, args=(coord, color, size, fobj, obj_angle))
    proc.start()
    if join:
        proc.join()
    return


def sample_coord_2d(coords, dist=10, add_corner_nz=True):
    total_coords = coords.shape[0]
    pick_coords_index = list(range(total_coords))
    if add_corner_nz:
        idx_ymin = np.argmin(coords[:, 0])
        idx_ymax = np.argmax(coords[:, 0])
        idx_xmin = np.argmin(coords[:, 1])
        idx_xmax = np.argmax(coords[:, 1])
        idx_xy1 = np.argmin(coords[:, 0] + coords[:, 1])
        idx_xy2 = np.argmax(coords[:, 0] + coords[:, 1])
        idx_xy3 = np.argmin(coords[:, 0] - coords[:, 1])
        idx_xy4 = np.argmax(coords[:, 0] - coords[:, 1])
        pick_coords_index = [idx_ymin, idx_ymax, idx_xmin, idx_xmax, idx_xy1, idx_xy2, idx_xy3, idx_xy4] + pick_coords_index
    pick_coords = coords[pick_coords_index].astype(float)

    tree = KDTree(pick_coords[:, :2])
    pick_idx = np.ones(len(pick_coords), dtype=bool)
    for i in range(len(pick_coords)):
        if pick_idx[i]:
            pick_idx[tree.query_ball_point(pick_coords[i, :2], dist)] = False
            pick_idx[i] = True # dist from itself is always 0
    pick_coords_sparse=pick_coords[pick_idx]

    return pick_coords_sparse, np.arange(total_coords)[pick_coords_index][pick_idx]


def pad_linear(x):
    y_shape, x_shape = x.shape
    x_pad=np.zeros((y_shape+2, x_shape+2))
    x_pad[1:-1, 1:-1]=x.copy()
    x_pad[0,:]=2*x_pad[1,:]-x_pad[2,:]
    x_pad[-1,:]=2*x_pad[-2,:]-x_pad[-3,:]
    x_pad[:,0]=2*x_pad[:,1]-x_pad[:,2]
    x_pad[:,-1]=2*x_pad[:,-2]-x_pad[:,-3]
    return x_pad


def numerical_diff_uv(z_mgrid, y_mgrid, x_mgrid):
    """given 3 2dArray xyz mgrids(vu), return norm vector zyx"""
    x_pad = pad_linear(x_mgrid)
    y_pad = pad_linear(y_mgrid)
    z_pad = pad_linear(z_mgrid)
    dxdu = (x_pad[1:-1, 2:] - x_pad[1:-1, 0:-2]) / 2
    dydu = (y_pad[1:-1, 2:] - y_pad[1:-1, 0:-2]) / 2
    dzdu = (z_pad[1:-1, 2:] - z_pad[1:-1, 0:-2]) / 2
    dxdv = (x_pad[2:, 1:-1] - x_pad[0:-2, 1:-1]) / 2
    dydv = (y_pad[2:, 1:-1] - y_pad[0:-2, 1:-1]) / 2
    dzdv = (z_pad[2:, 1:-1] - z_pad[0:-2, 1:-1]) / 2

    # z=z(u,v), y=z(u,v), x=z(u,v) -> vz,vy,vx = [dx/du*dy/dv-dy/du*dx/dv,dz/du*dx/dv-dx/du*dz/dv,dy/du*dz/dv-dz/du*dy/dv]
    v_zyx = np.array([dxdu*dydv-dydu*dxdv, dzdu*dxdv-dxdu*dzdv, dydu*dzdv-dzdu*dydv])
    v_zyx = v_zyx / np.sqrt(v_zyx[0]**2 + v_zyx[1]**2 + v_zyx[2]**2)

    return v_zyx


def generate_mgrid_rbf(zyxuv, thick, smooth=0, plot_process=False, fobj=None, obj_angle=0,
                       mean_filter=11, path_rbfcoord=None, interval=1, del_ratio=1/20):
    '''
    generate coords for new tomo that write in mgrid
    mgrid[:,i,j,k] is corresponding [z,y,x] in original tomo for point [i,j,k] in new tomo 
    thick is z depth above and under the plane/curve, you can sum tomo in z axis to get projection 
    '''

    z_data, y_data, x_data, u_data, v_data = zyxuv.T
    uv_data = np.stack([u_data,v_data], axis=1)
    xyz_data = np.stack([x_data, y_data, z_data], axis=1)
    spline_xyz = RBFInterpolator(uv_data, xyz_data, kernel='thin_plate_spline', smoothing=smooth)
    err = np.linalg.norm(spline_xyz(uv_data)-xyz_data, axis=1)
    print("num:",len(z_data),"smooth:",smooth,"rbf rmsd:",np.sqrt(err**2).mean())
    if smooth==0 and plot_process:
        coords_show,color,size=organize_coords([zyxuv[:, :3]],[(0,1,0)],[2])
        plot_mesh(coords_show,color,size,fobj,obj_angle)

    if smooth>0:
        n_bad=max(1, int(len(zyxuv)*del_ratio) ) # delete at least one point
        err_sort=err.argsort() # small to big
        zyxuv_good=zyxuv[err_sort][:len(zyxuv)-n_bad]
        if plot_process:
            zyxuv_bad=zyxuv[err_sort][len(zyxuv)-n_bad:]
            coords_show,color,size=organize_coords([zyxuv_good[:, :3],zyxuv_bad[:, :3]],[(0,1,0),(1,0,1)],[2,2])
            plot_mesh(coords_show,color,size,fobj,obj_angle)
        zyxuv = zyxuv_good
        z_data, y_data, x_data, u_data, v_data = zyxuv.T
        uv_data = np.stack([u_data,v_data], axis=1)
        xyz_data = np.stack([x_data, y_data, z_data], axis=1)
        spline_xyz = RBFInterpolator(uv_data, xyz_data, kernel='thin_plate_spline', smoothing=smooth)
        err = np.linalg.norm(spline_xyz(uv_data)-xyz_data, axis=1)
        print("num:",len(z_data),"smooth:",smooth,"rbf rmsd:",np.sqrt(err**2).mean())

    if path_rbfcoord is not None:
        np.savetxt(path_rbfcoord, np.stack([x_data, y_data, z_data]), fmt='%d')

    u_min, v_min = np.min(zyxuv[:, 3:5], axis=0) - interval
    u_max, v_max = np.max(zyxuv[:, 3:5], axis=0) + interval
    result_mgrid = np.mgrid[-1 * thick:thick + interval:interval, v_min:v_max + interval:interval, u_min:u_max + interval:interval].astype(float)
    vu_mgrid = result_mgrid[1:3, 0, :, :].copy()
    v_mgrid, u_mgrid = vu_mgrid
    v_shape, u_shape = v_mgrid.shape
    xyz_data = spline_xyz(np.stack([u_mgrid.flatten(), v_mgrid.flatten()], axis=1))
    x_mgrid = xyz_data[:, 0].reshape(v_shape, u_shape)
    y_mgrid = xyz_data[:, 1].reshape(v_shape, u_shape)
    z_mgrid = xyz_data[:, 2].reshape(v_shape, u_shape)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(z_mgrid,origin='lower')
    # plt.scatter(zyxuv[:, 3]-u_mgrid[0,0], zyxuv[:, 4]-v_mgrid[0,0],s=1,c='red')
    # plt.subplot(132)
    # plt.imshow(y_mgrid,origin='lower')
    # plt.scatter(zyxuv[:, 3]-u_mgrid[0,0], zyxuv[:, 4]-v_mgrid[0,0],s=1,c='red')
    # plt.subplot(133)
    # plt.imshow(x_mgrid,origin='lower')
    # plt.scatter(zyxuv[:, 3]-u_mgrid[0,0], zyxuv[:, 4]-v_mgrid[0,0],s=1,c='red')
    # plt.show()
    fvu = np.array([z_mgrid, y_mgrid, x_mgrid])
    nvu = numerical_diff_uv(z_mgrid, y_mgrid, x_mgrid)

    if mean_filter>0:
        mean_filter = int(mean_filter)//2*2+1  # odd number
        nvu = nd.uniform_filter(nvu, size=(1,mean_filter,mean_filter))
        nvu = nvu / np.sqrt(nvu[0]**2 + nvu[1]**2 + nvu[2]**2)

    i = 0
    for d in np.arange(-1 * thick, thick + interval, interval):
        result_mgrid[:, i, :, :] = fvu + d * nvu
        i += 1

    return result_mgrid, vu_mgrid


def interp_mgrid2tomo(mgrid, tomo_map, fill_value=None, order=1):
    '''
    value of point [i,j,k] in final tomo is the value of point mgrid[:,i,j,k] in original tomo
    if mgrid[:,i,j,k] out of range of origin, fill by mean
    '''
    if fill_value is None:
        fill_value=np.nan
    interp=map_coordinates(tomo_map,mgrid,order=order,prefilter=False,cval=fill_value,output=np.float32)
    if np.isnan(interp).any():
        interp[np.isnan(interp)]=interp[~np.isnan(interp)].mean()
    return interp


def Extract_Surface(obj_path,density_path,
                    output,convert_file,rbfcoord_file,
                    do_rotate       = True,
                    thick           = None,
                    fill_value      = None,
                    rbf_dist        = 0,
                    smooth_factor   = 3,
                    plot_process    = False,
                    filt            = False):

    smooth_factor=10**smooth_factor-1 # map 0 to 0

    zyxuv,angle=read_obj(obj_path,do_rotate)
    print('total coords num', len(zyxuv))
    if rbf_dist > 0:
        _, pick_idx = sample_coord_2d(zyxuv[:, 3:5], rbf_dist)
        zyxuv = zyxuv[pick_idx]
    print('final pick', len(zyxuv))

    with mrcfile.mmap(density_path) as mrc:
        tomo_density = mrc.data
        voxel_size = mrc.voxel_size

    print("start rbf...")
    mgrid_surf, vu_mgrid = generate_mgrid_rbf(zyxuv, thick,
                                    smooth=smooth_factor,
                                    plot_process=plot_process,
                                    mean_filter=11,
                                    path_rbfcoord=rbfcoord_file,
                                    fobj=obj_path,
                                    obj_angle=angle)

    # the ralationship between new tomo coords and origin tomo coords is stored in mgrid
    # [i,j,k] in new tomo correspond to mgrid[:,i,j,k]-->[z,y,x](float) in origin tomo
    y_top = mgrid_surf[1, mgrid_surf.shape[1] // 2, -1, mgrid_surf.shape[3] // 2]
    y_bot = mgrid_surf[1, mgrid_surf.shape[1] // 2, 0, mgrid_surf.shape[3] // 2]
    if y_top < y_bot:
        rotate180 = True
    else:
        rotate180 = False

    if filt:
        print("start filt...")
        _, vertex_uvs, triangles = read_obj_xyz_uv_triangle(obj_path)
        # mesh = o3d.io.read_triangle_mesh(obj_path)
        # trangles = np.array(mesh.triangles)
        # vertex_uvs = np.zeros((trangles.max()+1, 2))
        # vertex_uvs[trangles.flatten()] = np.array(mesh.triangle_uvs)
        rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vertex_uvs = np.dot(rot_matrix, vertex_uvs.T).T
        mask = ~triangle2d_mask(vertex_uvs, triangles, vu_mgrid)
        mgrid_surf[:, :, mask] -= (mgrid_surf[:, :, mask].max()+10)

    print("start interp...")
    if rotate180:
        mgrid_surf = mgrid_surf[:,:,::-1,::-1] # rotate 180
    if fill_value==0: fill_value=None # not a good input way
    final_tomo_surf = interp_mgrid2tomo(mgrid_surf, tomo_density, fill_value)

    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(final_tomo_surf)
        mrc.voxel_size = voxel_size

    np.save(convert_file, mgrid_surf.astype(np.float32))

    return


def main(args):
    fin:str=args.fin
    ftomo:str=args.ftomo
    fout:str=args.fout
    thick:int=args.thick
    rbf_dist:float=args.rbf_dist
    smooth:float=args.smooth
    id1:int=args.id1
    id2:int=args.id2
    show_3d:bool=args.show_3d
    filt:bool=args.filt
    man:bool=args.man

    pre = "manual" if man else "surface"

    os.makedirs(fout, exist_ok=True)
    output = os.path.join(fout, f'{pre}_{id1}-{id2}_RBF_{int(rbf_dist)}_thick_{thick}_result.mrc')
    convert_file = os.path.join(fout, f'{pre}_{id1}-{id2}_RBF_{int(rbf_dist)}_thick_{thick}_convert_coord.npy')
    rbfcoord_file = os.path.join(fout, f'{pre}_{id1}-{id2}_RBF_InterpCoords.txt')
    config_file = os.path.join(fout, f'{pre}_{id1}-{id2}.config')

    # write fake config
    config = configparser.ConfigParser()
    config.add_section("Parameter")
    config.set("Parameter", "Method", "RBF")
    config.set("Parameter", "RBFSample", str(rbf_dist))
    config.set("Parameter", "PolyOrder", "8")
    config.set("Parameter", "Thickness", str(thick))
    config.set("Parameter", "CylinderOrder", "0")
    config.set("Parameter", "FillValue", "0")
    config.set("Parameter", "smoothfactor",str(smooth))
    config.set("Parameter", "expandratio","0")
    with open(config_file, "w") as f:
        config.write(f)

    Extract_Surface(fin, ftomo, output, convert_file, rbfcoord_file,
                    thick=thick,
                    rbf_dist=rbf_dist,
                    smooth_factor=smooth,
                    plot_process=show_3d,
                    filt=filt,
                    do_rotate=True,
                    fill_value=None)
    
    print("finish")
    return


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fin', type=str, required=True,
                        help='input obj file with texture coordinate')
    parser.add_argument('--ftomo', type=str, required=True,
                        help='input raw mrc file')
    parser.add_argument('--fout', type=str, required=True,
                        help='output folder')
    parser.add_argument('--thick', type=int, default=10,
                        help='thick above and under the surface, in pixel')
    parser.add_argument('--rbf_dist', type=float, default=5,
                        help='use rbf interpolation with this minimum sample distance in pixel')
    parser.add_argument('--smooth', type=float, default=3,
                        help='smooth factor of RBF')
    parser.add_argument('--id1', type=int, default=2,
                        help='output id')
    parser.add_argument('--id2', type=int, default=2,
                        help='output id')
    parser.add_argument('--show_3d', action='store_true',
                        help='only used when flatten')
    parser.add_argument('--filt', action='store_true',
                        help='filt the part out of the input mesh')
    parser.add_argument('--man', action='store_true',
                        help='output file start with manual_ (defaul surface_)')
    args = parser.parse_args()

    main(args)