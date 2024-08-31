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

import numpy as np
import mrcfile
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
import argparse
import tqdm
from multiprocessing import Pool, set_start_method
from pathlib import Path


def interp_mgrid2tomo(mgrid, tomo_map, fill_value=None, order=1):
    '''from mpicker_core'''
    if fill_value is None:
        fill_value=np.nan
    interp=map_coordinates(tomo_map,mgrid,order=order,prefilter=False,cval=fill_value,output=np.float32)
    if np.isnan(interp).any():
        interp[np.isnan(interp)]=interp[~np.isnan(interp)].mean()
    return interp


def ctfshift_warp(ctf):
    sz, sy, sx = ctf.shape
    if sz//2 >= sy:
        ctf = ctf[:sy] # Relion4
        sz, sy, sx = ctf.shape
    result = np.zeros((sz,sz,sz), dtype=np.float32)
    result[:, :, 0:sx] = ctf
    result[1:, 1:, sx:sz] = ctf[1:, 1:, 1:sz-sx+1][::-1, ::-1, ::-1]
    result[0, 1:, sx:sz] = ctf[0, 1:, 1:sz-sx+1][::-1, ::-1]
    result[1:, 0, sx:sz] = ctf[1:, 0, 1:sz-sx+1][::-1, ::-1]
    result[0, 0, sx:sz] = ctf[0, 0, 1:sz-sx+1][::-1]
    result = np.fft.fftshift(result)
    return result


def read_file(fpath):
    with open(fpath, 'r') as f:
        data=f.readlines()
    result = []
    for line in data:
        line = line.strip().split()
        if len(line) < 1 or line[0][0] == "#":
            continue
        else:
            result.append(line[0])
    return result


def load_file(fpath, length):
    if len(fpath) > 3 and fpath[-3:] == "txt":
        data = read_file(fpath)
    else:
        data = [fpath for _ in range(length)]
    if len(data) != length:
        raise Exception("number of file in .txt should equal to number of particles.")
    return data


def give_matrix(abc, use_vector=False):
    # relion euler angle
    if use_vector:
        n_vector = abc / np.linalg.norm(abc)
        tilt = np.arccos(n_vector[2])
        psi = np.arctan2(n_vector[1], -n_vector[0])
        rot = 0
    else:
        rot, tilt, psi = abc * np.pi / 180
    rot, tilt, psi = -rot, -tilt, -psi
    try:
        rot_matrix = R.from_euler('zyz', [rot, tilt, psi]).as_matrix()  # as_dcm() in old scipy
    except:
        rot_matrix = R.from_euler('zyz', [rot, tilt, psi]).as_dcm()
    return rot_matrix


def convert_back_coord(coord, abc, use_vector=False):
    __, nz, ny, nx = coord.shape
    coord = coord.reshape((3, nz * ny * nx)).transpose()  #convert to [[z1,y1,x1],[z2,y2,x2]...]
    convert_matrix = give_matrix(abc, use_vector)
    coord_xyz = np.flip(coord.transpose(), axis=0) # .astype(float)
    coord_xyz_convert = np.dot(convert_matrix, coord_xyz)
    return np.flip(coord_xyz_convert, axis=0).reshape((3, nz, ny, nx))  #convert back


def interp_mgrid(x0, y0, z0, dxy, dz):
    dxy = int(dxy)
    dz = int(dz)
    # mgrid = np.mgrid[-(dz - 1) / 2:(dz + 1) / 2, -(dxy - 1) / 2:(dxy + 1) / 2, -(dxy - 1) / 2:(dxy + 1) / 2]
    mgrid = np.mgrid[-(dz-1)/2:(dz+1)/2, -(dxy//2):dxy-(dxy//2), -(dxy//2):dxy-(dxy//2)] # let size//2+1 be center of image
    trans = np.zeros((3,1,1,1))
    trans[0] = z0 - 1
    trans[1] = y0 - 1
    trans[2] = x0 - 1
    return trans, mgrid


def project_particle(tomo_map, x0, y0, z0, dxy, dz, nx, ny, nz, use_vector=False, fill_value=None, shiftz=0, order=1):
    # nx,ny,nz are rot tilt psi when use_vector is False
    # xyz start from 1, trans start from 0
    trans, mgrid = interp_mgrid(x0, y0, z0, dxy, dz)
    mgrid[0] += shiftz
    mgrid = convert_back_coord(mgrid, np.array([nx, ny, nz]), use_vector) + trans
    tomo = interp_mgrid2tomo(mgrid, tomo_map, fill_value, order)
    return tomo.sum(axis=0)


def worker_func(args):
    """
    map_path, x0, y0, z0, dxy, dz, nx, ny, nz, use_vector, shiftz, order \n
    fill_value=None
    """
    map_path, x0, y0, z0, dxy, dz, nx, ny, nz, use_vector, shiftz, order = args
    with mrcfile.mmap(map_path) as mrc:
        tomo = mrc.data
    return project_particle(tomo, x0, y0, z0, dxy, dz, nx, ny, nz, use_vector, None, shiftz, order)

def worker_func_tomo(args):
    """
    similiar to worker_func, but do not project volume
    """
    map_path, x0, y0, z0, dxy, dz, nx, ny, nz, use_vector, shiftz, order = args
    with mrcfile.mmap(map_path) as mrc:
        tomo = mrc.data
    trans, mgrid = interp_mgrid(x0, y0, z0, dxy, dz)
    mgrid[0] += shiftz
    mgrid = convert_back_coord(mgrid, np.array([nx, ny, nz]), use_vector) + trans
    tomo_rotate = interp_mgrid2tomo(mgrid, tomo, None, order)
    return tomo_rotate

def worker_func_ctf(args):
    """
    map_path, x0, y0, z0, dxy, nx, ny, nz, use_vector, flipmap, shiftmap, order \n
    dz=1, fill_value=0, shiftz=0
    """
    map_path, x0, y0, z0, dxy, nx, ny, nz, use_vector, flipmap, shiftmap, order = args
    with mrcfile.mmap(map_path) as mrc:
        tomo = mrc.data
    if shiftmap:
        tomo = ctfshift_warp(tomo)
    if flipmap:
        tomo = tomo[::-1]
        if x0 is not None: # user provide xyz
            z0 = tomo.shape[0] + 1 - z0
    if x0 is None:
        x0 = tomo.shape[2] // 2 + 1
        y0 = tomo.shape[1] // 2 + 1
        z0 = tomo.shape[0] // 2 + 1
    return project_particle(tomo, x0, y0, z0, dxy, 1, nx, ny, nz, use_vector, 0, 0, order)


def generate_thu(fmap, fctf, fout, num):
    fmap = Path(fmap).name
    fctf = Path(fctf).name
    text_0_6 = "1 " * 7
    text_9_26 = "1 " * 18
    result = []
    for i in range(1, num+1):
        line = f"{text_0_6}{i:06d}@{fmap} none {text_9_26}{i:06d}@{fctf} \n"
        result.append(line)
    with open(fout, mode="w") as f:
        f.writelines(result)


def main(args):
    data_path = args.data
    dxy = args.dxy
    dz = args.dz
    use_vector = args.use_vector
    max_workers = args.process
    shiftz = args.shiftz
    flip_ctf = args.ctfflip
    shift_ctf = args.ctfshift
    interp_order = 1
    invert = args.invert
    output = args.output
    ctf_output = args.ctfout
    tomo_output = args.tomoout
    fthu = args.thu

    coord_angle = np.loadtxt(data_path, ndmin=2, dtype=float)  # x y z nx ny nz (dx dy dz)
    if coord_angle.shape[1] >= 9:
        coord_angle[:, 0:3] -= coord_angle[:, 6:9] # realX = X - originX
    coord_angle = coord_angle[:, 0:6]
        
    if args.map is not None:
        doparticle = True
        map_path = args.map
        map_path = load_file(map_path, len(coord_angle))
        with mrcfile.mmap(map_path[0]) as mrc:
            voxel_size = mrc.voxel_size
        shiftz = load_file(shiftz, len(coord_angle))
        shiftz = [float(sz) for sz in shiftz]
        result = []
    else:
        doparticle = False
        voxel_size = None

    if args.ctf is not None:
        doctf = True
        ctf_map_path = args.ctf
        ctf_map_path = load_file(ctf_map_path, len(coord_angle))
        if args.ctfcenter is None:
            ctf_x, ctf_y, ctf_z = None, None, None
        else:
            ctf_x, ctf_y, ctf_z = [float(i) for i in args.ctfcenter.split(',')]
        ctf_result = []
    else:
        doctf = False

    if args.map is not None and tomo_output is not None:
        dotomo = True
    else:
        dotomo = False

    with Pool(processes=max_workers) as pool:
        if doparticle:
            input_args = [(mpath, x0, y0, z0, dxy, dz, nx, ny, nz, use_vector, sz, interp_order) 
                          for mpath, sz, (x0, y0, z0, nx, ny, nz) in zip(map_path, shiftz, coord_angle)]
            for res in tqdm.tqdm(pool.imap(worker_func, input_args), total=len(input_args), desc="extract particle"):
                result.append(res)
        if doctf:
            input_args = [(ctf_mpath, ctf_x, ctf_y, ctf_z, dxy, nx, ny, nz, use_vector, flip_ctf, shift_ctf, interp_order) 
                          for ctf_mpath, (x0, y0, z0, nx, ny, nz) in zip(ctf_map_path, coord_angle)]
            for res in tqdm.tqdm(pool.imap(worker_func_ctf, input_args), total=len(input_args), desc="extract 2dctf"):
                ctf_result.append(res)

    if doparticle:
        result = np.array(result, dtype=np.float32)
        if invert:
            result *= -1
        with mrcfile.new(output, overwrite=True) as mrc:
            mrc.set_data(result)
            if voxel_size is not None:
                mrc.voxel_size = voxel_size
            mrc.set_image_stack()
    if doctf:
        with mrcfile.new(ctf_output, overwrite=True) as mrc:
            mrc.set_data(np.array(ctf_result, dtype=np.float32))
            if voxel_size is not None:
                mrc.voxel_size = voxel_size
            mrc.set_image_stack()

    if fthu is not None:
        if output is not None and ctf_output is not None:
            generate_thu(output, ctf_output, fthu, len(coord_angle))
        else:
            print("provide both output and ctfout to generate .thu file")

    if dotomo:
        # let dz=dxy here
        input_args = [(mpath, x0, y0, z0, dxy, dxy, nx, ny, nz, use_vector, sz, interp_order) 
                          for mpath, sz, (x0, y0, z0, nx, ny, nz) in zip(map_path, shiftz, coord_angle)]
        sta = np.zeros((dxy,dxy,dxy), dtype=float)
        for args in tqdm.tqdm(input_args, desc="STA"):
            sta += worker_func_tomo(args)
        sta = sta/len(args)
        if invert:
            sta *= -1
        with mrcfile.new(tomo_output, overwrite=True) as mrc:
            mrc.set_data(sta.astype(np.float32))
            if voxel_size is not None:
                mrc.voxel_size = voxel_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract 2d particle projection on membrane.")
    parser.add_argument('--map', type=str,
                        help='tomogram file(or a .txt file of tomos list)')
    parser.add_argument('--data', type=str, required=True,
                        help='file contains x,y,z,rot,tilt,psi, xyz start from 1 (in pixel), angels same as relion. or x,y,z,rot,tilt,psi,dx,dy,dz (rlnOriginXYZ)')
    parser.add_argument('--dxy', type=int, default=50,
                        help='side length of output images, in pixel')
    parser.add_argument('--dz', type=int, default=1,
                        help='depth to project (thick), in pixel, default 1')
    parser.add_argument('--output', type=str, default='2d_output.mrcs',
                        help='output file, default 2d_output.mrcs')
    parser.add_argument('--shiftz', type=str, default="0",
                        help='shift the center along z when project(or a .txt file of shiftzs list), in pixel.')
    parser.add_argument('--invert', action='store_true',
                        help='invert the result projection, black to white')
    parser.add_argument('--ctf', type=str,
                        help='can input a 3dctf(or a .txt file of 3dctfs list) to generate 2dctfs. should have same size as dxy')
    parser.add_argument('--ctfout', type=str, default='2dctf_output.mrcs',
                        help='name of 2dctf file, default 2dctf_output.mrcs')
    parser.add_argument('--ctfcenter', type=str,
                        help='xyz center of 3dctf, start from 1, e.g "10,10,10", default use size//2+1')
    parser.add_argument('--ctfflip', action='store_true',
                        help='flipz for the 3dctf at first, useful when 3dctf generated by Relion2/3.')
    parser.add_argument('--ctfshift', action='store_true',
                        help='useful when 3dctf is generated by Warp or Relion4.')
    parser.add_argument('--thu', type=str,
                        help='generate .thu file for class2D in THUNDER2 if provided')
    parser.add_argument('--tomoout', type=str,
                        help='sum all subtomo to this mrc file if provided, might be slow.')
    parser.add_argument('--use_vector', action='store_true',
                        help='provide norm vector nx ny nz in data, instead of rot tilt psi. same as rot=0')
    parser.add_argument('--process', type=int, default=1,
                        help='use multi process. not too large, because NumPy used multi thread already.')
    args = parser.parse_args()
    set_start_method('spawn')
    main(args)