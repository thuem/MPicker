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

import argparse, mrcfile
import numpy as np
import os

# in .npz, we use zyx and start from 0.

def write_surface_npz(size_zyx, coords, filename, write_mrc=False):
    ''' coords is [[zyx],[zyx],...] same as output of np.argwhere '''
    if coords.ndim == 1:
        coords = np.expand_dims(coords, axis=0)
    coords = coords.astype(int)
    if len(coords) > 0 and len(coords[0]) > 0:
        mask = (coords[:, 0] < size_zyx[0]) & (coords[:, 0] >=0) \
            & (coords[:, 1] < size_zyx[1]) & (coords[:, 1] >=0) \
            & (coords[:, 2] < size_zyx[2]) & (coords[:, 2] >=0)
        coords = coords[mask]
    if write_mrc:
        surface = np.zeros(size_zyx, dtype=np.int8)
        if coords.ndim == 2 and len(coords) > 0 and len(coords[0]) > 0:
            surface[coords.T[0], coords.T[1], coords.T[2]] = 1
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(surface.astype(np.int8))
        return
    else:
        np.savez(filename, size_zyx=size_zyx, coords=coords)
        if not filename.endswith(".npz"):
            os.rename(filename+".npz", filename)
        return


def read_surface_coord(filename, writenpz=True):
    '''input npz or mrc return coords '''
    if not os.path.exists(filename):
        filename = filename.replace(".mrc.npz",".mrc")
        nofile = True
    else:
        nofile = False
    try:
        coords = np.load(filename)['coords']
        if coords.ndim == 1:
            coords = np.expand_dims(coords, axis=0)
    except:
        with mrcfile.mmap(filename) as mrc:
            surf = mrc.data
        coords = np.argwhere(surf)
        if writenpz and nofile:
            try:
                np.savez(filename+".npz", size_zyx=surf.shape, coords=coords.astype(int))
                print("conver",filename,"to",filename+".npz")
            except:
                print("fail to write", filename+".npz")
    return coords


def read_surface_mrc(filename, dtype=np.int8):
    '''input npz or mrc return ndarray '''
    if not os.path.exists(filename):
        filename = filename.replace(".mrc.npz",".mrc")
    try:
        file = np.load(filename)
        size = file['size_zyx']
        coords = file['coords']
        surface = np.zeros(size, dtype=dtype)
        if coords.ndim == 2 and len(coords) > 0 and len(coords[0]) > 0:
            surface[coords.T[0], coords.T[1], coords.T[2]] = 1
    except:
        with mrcfile.mmap(filename) as mrc:
            surface = mrc.data.astype(dtype)
    return surface


def coords2image(coords, size_yx, z, inverty=True):
    '''coords is zyx start from 0, z start from 0'''
    sy, sx = int(size_yx[0]), int(size_yx[1])
    image=np.zeros((sy, sx), dtype=bool)
    if coords.ndim == 1:
        coords = np.expand_dims(coords, axis=0)
    coords = coords.astype(int)
    if len(coords) > 0 and len(coords[0]) > 0:
        coords = coords[coords[:,0]==int(z)][:,1:]
        coords = coords[(coords[:,0]>=0)&(coords[:,1]>=0)&(coords[:,0]<sy)&(coords[:,1]<sx)]
        if len(coords) > 0:
            image[coords.T[0], coords.T[1]] = 1
        if inverty:
            image = image[::-1,:]
    return image


def main(args):
    npz_name = args.npz
    mrc_name = args.mrc
    out_name = args.out
    if npz_name is not None:
        data = read_surface_mrc(npz_name)
        with mrcfile.new(out_name, overwrite=True) as mrc:
            mrc.set_data(data)
        return
    elif mrc_name is not None:
        with mrcfile.mmap(mrc_name) as mrc:
            surf = mrc.data
            coords = np.argwhere(surf)
            size = surf.shape
        write_surface_npz(size, coords, out_name)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert npz to mrc, or mrc to npz")
    parser.add_argument('--npz', type=str, help='input name of npz file')
    parser.add_argument('--mrc', type=str, help='input name of mrc file')
    parser.add_argument('--out', type=str, help='output file name')
    args = parser.parse_args()
    main(args)