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

import configparser, argparse, os
from mpicker_core import Find_Surface, Extract_Surface


def get_str_list(string):
    try:
        str_list = eval(string)
        if type(str_list) is not list:
            str_list = string
    except:
        str_list = string
    return str_list


def Init_config_surface(ini_config_path, surf_config_path):
    Surface_name = os.path.basename(os.path.dirname(surf_config_path))
    config_main = configparser.ConfigParser()
    config_main.read(ini_config_path, encoding='utf-8')
    if config_main.has_section("Path"):
        if config_main.get('Path', 'Surface', fallback='None') == 'None':
            config_main.set('Path', 'Surface', Surface_name)
        else:
            same_flag = False
            Surface_string = config_main.get('Path', 'Surface')
            for Surface in Surface_string.split():
                if Surface ==  Surface_name:
                    same_flag = True
            if same_flag == False:
                Surface_string = " ".join(sorted( Surface_string.split(), key=lambda x: int(x.split("_")[1]) ))
                Surface_string = Surface_string + " " +  Surface_name
                config_main.set('Path', 'Surface', Surface_string)
    with open(ini_config_path, "w") as f:
        config_main.write(f)


def Find_Surface_gui(main_config_path, surf_config_path):
    # absolute path
    config_main = configparser.ConfigParser()
    config_main.read(main_config_path, encoding='utf-8')
    config_surf = configparser.ConfigParser()
    config_surf.read(surf_config_path, encoding='utf-8')

    if 'txt_path' in config_surf['Parameter'].keys():
        return

    near_ero = config_surf['Parameter']['NearEro']
    boundaryout_path = os.path.join(os.path.dirname(config_main['Path']['inputboundary']) , 'my_boundary_' + near_ero + '.mrc')
    if os.path.isfile(boundaryout_path):
        boundary_path = boundaryout_path
    else:
        boundary_path = None
    mask_path = config_main['Path']['inputmask']
    initial_point = eval(config_surf['Parameter']['Points']) # 2d list of number
    surfout_path = surf_config_path.replace(".config","_surf.mrc.npz")
    left2right = []
    for direction in get_str_list(config_surf['Parameter']['DirectionL2R']): 
        if direction == "Left To Right":
            left2right.append(True)
        else:
            left2right.append(False)
    xyz = get_str_list(config_surf['Parameter']['Facexyz'])
    surf_method = get_str_list(config_surf['Parameter']['mode'])
    min_surf = int(config_surf['Parameter']['minsurf'])
    elongation_pixel = eval(config_surf['Parameter']['maxpixel']) # int. can be list of int too
    n_cpu = int(config_surf['Parameter']['ncpu'])

    Find_Surface(
        boundary_path   = boundary_path,
        mask_path       = mask_path,
        initial_point   = initial_point,
        surfout_path    = surfout_path,
        boundaryout_path= boundaryout_path,
        near_ero        = near_ero,
        left2right      = left2right,
        xyz             = xyz,
        surf_method     = surf_method,
        min_surf        = min_surf,
        elongation_pixel= elongation_pixel,
        n_cpu           = n_cpu
        )
    

def Extract_Surface_gui(main_config_path, surf_config_path, plot_process, plot_fitting, print_time):
    # absolute path
    config_main = configparser.ConfigParser()
    config_main.read(main_config_path, encoding='utf-8')
    config_surf = configparser.ConfigParser()
    config_surf.read(surf_config_path, encoding='utf-8')
    surf_dir = os.path.dirname(surf_config_path)
    surf_prefix = os.path.basename(surf_config_path).replace('.config','')
    density_path = config_main['Path']['inputraw']

    surf_path = os.path.join(surf_dir, surf_prefix.split('-')[0] + '_surf.mrc.npz')
    surf_path_old = os.path.join(surf_dir, surf_prefix.split('-')[0] + '_surf.mrc')
    coordsurf = os.path.join(surf_dir, surf_prefix.split('-')[0] + '_surf.txt')
    if os.path.isfile(surf_path) or os.path.isfile(surf_path_old):
        coordsurf = None
    elif os.path.isfile(coordsurf):
        surf_path = None

    method = config_surf.get('Parameter', 'Method')
    rbf_dist = config_surf.getint('Parameter', 'RBFSample')
    order = config_surf.getint('Parameter', 'PolyOrder')
    thick = config_surf.getint('Parameter', 'Thickness')
    cylinder_order = config_surf.getint('Parameter', 'CylinderOrder')
    fill_value = config_surf.getfloat('Parameter', 'FillValue')
    smooth_factor = config_surf.getfloat('Parameter', 'smoothfactor')
    expand_ratio = config_surf.getfloat('Parameter', 'expandratio', fallback=0)

    if method == 'RBF':
        num = rbf_dist
        order = None
    else:
        num = order
        rbf_dist = None
    output = os.path.join(surf_dir, f'{surf_prefix}_{method}_{num}_thick_{thick}_result.mrc')
    convert_file = os.path.join(surf_dir, f'{surf_prefix}_{method}_{num}_thick_{thick}_convert_coord.npy')
    path_rbfcoord = os.path.join(surf_dir, f'{surf_prefix}_RBF_InterpCoords.txt')

    if cylinder_order == -1:
        # closed ellipse cylinder
        do_rotate = False
    else:
        do_rotate = True
    
    Extract_Surface(
        surf_path       = surf_path,
        density_path    = density_path,
        coordsurf       = coordsurf,
        output_plane    = None,
        output          = output,
        convert_file    = convert_file,
        cylinder_order  = cylinder_order,
        rbf_dist        = rbf_dist,
        order           = order,
        do_rotate       = do_rotate,
        thick           = thick,
        fill_value      = fill_value,
        smooth_factor   = smooth_factor,
        plot_process    = plot_process,
        plot_fitting    = plot_fitting,
        expand_ratio    = expand_ratio,
        print_time      = print_time,
        path_rbfcoord   = path_rbfcoord
        )
    print("finish")


def main(args):
    main_config_path = os.path.abspath(args.config_tomo)
    surf_config_path = os.path.abspath(args.config_surf)
    if args.mode == 'surffind':
        Init_config_surface(main_config_path, surf_config_path) # update main config
        Find_Surface_gui(main_config_path, surf_config_path) # read config and run
    elif args.mode == 'flatten':
        Extract_Surface_gui(main_config_path, surf_config_path, 
                            args.show_3d, args.show_fit, args.time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['surffind', 'flatten'],
                        help='task you want to do')
    parser.add_argument('--config_tomo', type=str,
                        help='config file of the tomo')
    parser.add_argument('--config_surf', type=str,
                        help='config file of the task. should in different folder for different surface.')
    parser.add_argument('--show_3d', action='store_true',
                        help='only used when flatten')
    parser.add_argument('--show_fit', action='store_true',
                        help='only used when flatten')
    parser.add_argument('--time', action='store_true',
                        help='only used when flatten')
    args = parser.parse_args()

    main(args)