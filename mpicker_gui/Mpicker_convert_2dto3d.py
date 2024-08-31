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
from scipy.spatial.transform import Rotation
import numpy as np


def process_3d(xyzrottiltpsidxdydz, use_vector=False):
    """return x,y,z,rot,tilt,psi,dx,dy,dz (in radian)"""
    if xyzrottiltpsidxdydz.shape[1] >= 9:
        xyzrottiltpsidxdydz = xyzrottiltpsidxdydz[:, :9]
    else:
        xyzrottiltpsidxdydz = np.insert(xyzrottiltpsidxdydz[:, :6], 6, np.zeros((3, 1)), axis=1)
    if not use_vector:
        xyzrottiltpsidxdydz[:, 3:6] *= (np.pi / 180)
        return xyzrottiltpsidxdydz
    x, y, z, nx, ny, nz, dx, dy, dz = xyzrottiltpsidxdydz.T
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    tilt = np.arccos(nz)
    psi = np.arctan2(ny, -nx)
    rot = np.zeros_like(psi)
    xyzrottiltpsidxdydz[:, 3:6] = np.column_stack((rot, tilt, psi))
    return xyzrottiltpsidxdydz


def process_2d(q0q1xy, relion=False, d_x=0, d_y=0, d_psi=0):
    """recenter and return angle,dx,dy (in radian)"""
    cls_id = None
    if relion:
        psi, dx, dy = q0q1xy.T[:3]
        psi *= (np.pi / 180)
        if q0q1xy.shape[1] > 3:
            cls_id = q0q1xy[:, 3].astype(int)
    else:
        q0, q1, dx, dy = q0q1xy.T[:4]
        psi = np.arctan2(q1, q0)  # strange, not 2*atan2
        dx, dy = -dx, -dy
        if q0q1xy.shape[1] > 4:
            cls_id = q0q1xy[:, 4].astype(int)
    # recenter
    dx += np.cos(psi) * d_x + np. sin(psi) * d_y
    dy += -np.sin(psi) * d_x + np.cos(psi) * d_y
    psi = psi + d_psi * (np.pi / 180)
    return np.column_stack((psi, dx, dy)), cls_id


def get_result(psixy, xyzrottiltpsidxdydz):
    """return x,y,z,rot,tilt,psi,dx,dy,dz (in degree), xyz not used in fact\n
       will change xyzrottiltpsidxdydz in place"""
    ang, dx, dy = psixy.T
    angles = xyzrottiltpsidxdydz[:, 3:6]
    matrix = Rotation.from_euler('zyz', -angles).as_matrix()
    coordinates = np.column_stack((dx, dy, np.zeros_like(dx)))
    xyzrottiltpsidxdydz[:, 6:9] += np.matmul(matrix, coordinates[..., np.newaxis])[:, :, 0]
    xyzrottiltpsidxdydz[:, 3] += ang
    
    xyzrottiltpsidxdydz[:, 3:6] *= (180 / np.pi)
    rot = xyzrottiltpsidxdydz[:, 3] % 360
    rot[rot > 180] -= 360 # -180 to +180
    xyzrottiltpsidxdydz[:, 3] = rot
    return xyzrottiltpsidxdydz


def main(args):
    data = np.loadtxt(args.data, ndmin=2)
    data_proj = np.loadtxt(args.data_proj, ndmin=2)
    if len(data) == 0:
        raise Exception("input data is empty!")
    if len(data) != len(data_proj):
        raise Exception(f"data and data_proj should have same number of lines. {len(data)} vs {len(data_proj)}")
        
    if data_proj.shape[1] < 6:
        raise Exception("data_proj less than 6 columns!")
    if args.use_star:
        if data.shape[1] < 3:
            raise Exception("data(star) less than 3 columns!")
    else:
        if data.shape[1] < 4:
            raise Exception("data(thu) less than 4 columns!")
        
    data, cls_id = process_2d(data, args.use_star, args.movex, args.movey, args.rotate)
    if cls_id is not None and args.cls >= 0:
        mask = cls_id==args.cls
        data = data[mask]
        data_proj = data_proj[mask]
        if len(data) == 0:
            raise Exception("input data is empty after specify class id")
    data_proj = process_3d(data_proj, args.use_vector_proj)
    result = get_result(data, data_proj)
    header = "rlnCoordinateX rlnCoordinateY rlnCoordinateZ rlnAngleRot rlnAngleTilt rlnAnglePsi rlnOriginX rlnOriginY rlnOriginZ"
    np.savetxt(args.output, result, header=header, fmt="%10.4f")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert class2D result to data for 3d refine")
    parser.add_argument('--data', type=str, required=True,
                        help='file contains Quat0,Quat1,TX,TY from thunder .thu file. last column can be class id')
    parser.add_argument('--data_proj', type=str, required=True,
                        help='file contains x,y,z,rot,tilt,psi(,dx,dy,dz), same as used in Mpicker_2dproject')
    parser.add_argument('--output', type=str, required=True,
                        help='output file, contains x,y,z,rot,tilt,psi,dx,dy,dz that can be used for relion')
    parser.add_argument('--use_vector_proj', action='store_true',
                        help='if use nx,ny,nx in data_proj, same as used in Mpicker_2dproject')
    parser.add_argument('--use_star', action='store_true',
                        help='if input data file in relion style, psi,originx,originy')
    parser.add_argument('--movex', type=float, default=0,
                        help='move class2D result for recenter, in pixel, default 0')
    parser.add_argument('--movey', type=float, default=0,
                        help='move class2D result for recenter, in pixel, default 0')
    parser.add_argument('--rotate', type=float, default=0,
                        help='rotate class2D result, after recenter, in degree, default 0')
    parser.add_argument('--cls', type=int, default=-1,
                        help='can specify a class id to convert if >=0, default all')
    
    args = parser.parse_args()
    main(args)