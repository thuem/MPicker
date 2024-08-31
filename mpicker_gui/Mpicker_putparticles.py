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
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import argparse


def rotation_mgrid(rot, tilt, psi, dx, dy, dz, size, relion=True):
    # rotate and then shift
    # output mgrid that mgrid[:,z,y,x] is zyx
    if relion:
        rot, tilt, psi = -rot, -tilt, -psi
    try:
        rot_matrix = Rotation.from_euler('zyz', [rot, tilt, psi], degrees=True).as_dcm()
    except:
        rot_matrix = Rotation.from_euler('zyz', [rot, tilt, psi], degrees=True).as_matrix()
    rot_matrix = rot_matrix.T  # inverse
    mgrid = np.mgrid[0:size, 0:size, 0:size]
    center = size // 2  # star from 0 here
    mgrid -= center
    mgrid.resize((3, size**3))
    mgrid = np.dot(rot_matrix, mgrid)
    mgrid.resize((3, size, size, size))
    shiftxyz = center + np.array([dx, dy, dz]).reshape((3, 1, 1, 1))
    mgrid += shiftxyz
    return mgrid.transpose((0, 3, 2, 1))[::-1]


def rotate_protein(protein_tomo, rot, tilt, psi, dx, dy, dz, relion=True):
    # xyz start from 0, shape xyz of protein should be same
    len_pro = protein_tomo.shape[0]
    mgrid = rotation_mgrid(rot, tilt, psi, dx, dy, dz, len_pro, relion)
    protein_rotate = map_coordinates(protein_tomo, mgrid, order=1, cval=0)
    return protein_rotate


def insert_proteins(tomo, protein_tomo, coord_ori_shift, relion=True):
    sz, sy, sx = tomo.shape
    len_pro = protein_tomo.shape[0]
    cen_pro = len_pro // 2  # start from 0
    for x, y, z, rot, tilt, psi, dx, dy, dz in tqdm(coord_ori_shift, desc="put particles"):
        if relion:
            dx, dy, dz = -dx, -dy, -dz
        x, y, z = x + dx, y + dy, z + dz
        cx, cy, cz = int(round(x)), int(round(y)), int(round(z))  # int part
        dx, dy, dz = x - cx, y - cy, z - cz  # float part
        rot_protein = rotate_protein(protein_tomo, rot, tilt, psi, dx, dy, dz, relion)
        if rot_protein.dtype != tomo.dtype:
            rot_protein = rot_protein.astype(tomo.dtype)
        x_left, y_left, z_left = min(cx, cen_pro), min(cy, cen_pro), min(cz, cen_pro)
        x_right, y_right, z_right = min(len_pro - 1 - cen_pro, sx - 1 - cx), min(len_pro - 1 - cen_pro, sy - 1 - cy), min(len_pro - 1 - cen_pro, sz - 1 - cz)
        if x_left + x_right < 0 or y_left + y_right < 0 or z_left + z_right < 0:
            continue
        tslice_x, tslice_y, tslice_z = slice(cx - x_left, cx + x_right + 1), slice(cy - y_left, cy + y_right + 1), slice(cz - z_left, cz + z_right + 1)
        pslice_x, pslice_y, pslice_z = slice(cen_pro - x_left, cen_pro + x_right + 1), slice(cen_pro - y_left, cen_pro + y_right + 1), slice(cen_pro - z_left, cen_pro + z_right + 1)
        tomo[tslice_z, tslice_y, tslice_x] += rot_protein[pslice_z, pslice_y, pslice_x]
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="put particles back in tomo")
    parser.add_argument('--protein', type=str, required=True,
                        help='the mrc file of the particle, should be cubic')
    parser.add_argument('--data', type=str, required=True,
                        help='file contains x,y,z,rot,tilt,psi, xyz start from 1 (in pixel), angels same as relion. or x,y,z,rot,tilt,psi,dx,dy,dz (rlnOriginXYZ)')
    parser.add_argument('--tomo', type=str,
                        help='the background tomo to add particles in place')
    parser.add_argument('--xyz', type=str,
                        help='generate a new background tomo with shape xyz, for example "300,300,100"')
    parser.add_argument('--out', type=str,
                        help='output name if generate new tomo')
    args = parser.parse_args()


    data_file = args.data
    protein_file = args.protein
    data = np.loadtxt(data_file, ndmin=2)
    with mrcfile.open(protein_file, permissive=True) as mrc:
        protein = mrc.data

    if protein.shape[0] == protein.shape[1] == protein.shape[2]:
        pass
    else:
        raise Exception("protein should be cubic (shape x=y=z)")

    if data.shape[1] < 6:
        raise Exception("data file should have at least 6 columns")
    elif data.shape[1] >= 9:
        data = data[:, :9]
    else:
        coord_ori = data[:, :6]
        data = np.zeros((data.shape[0], 9), dtype=float)
        data[:, :6] = coord_ori

    if args.tomo is not None:
        tomo_file = args.tomo
    elif args.xyz is not None and args.out is not None:
        sx, sy, sz = [ int(s) for s in args.xyz.split(",") ]
        tomo_file = args.out
        print("generate background tomo")
        with mrcfile.new_mmap(tomo_file, overwrite=True, shape=(sz, sy, sx), mrc_mode=2, fill=0) as mrc:
            tomo = mrc.data
    else:
        raise Exception("should provide tomo, or provide xyz and out")

    with mrcfile.mmap(tomo_file, mode='r+', permissive=True) as mrc:
        tomo = mrc.data
        insert_proteins(tomo, protein, data, relion=True)

