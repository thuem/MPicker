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
from scipy.spatial import KDTree
import argparse


def main_local2global(coords, mgrid, outvector=False):
    out_coords = []
    out_vectors = []
    ni,nj,nk = mgrid.shape[1:]
    for x, y, z in coords:
        i, j, k = int(round(z)-1), int(round(y)-1), int(round(x)-1)
        if i>=0 and j>=0 and k>=0 and i<ni and j<nj and k<nk:
            gi, gj, gk = mgrid[:, i, j, k]
            if gi>=0 and gj>=0 and gk>=0:
                gx, gy, gz = gk+1, gj+1, gi+1
                out_coords.append([x, y, z, gx, gy, gz])
                if outvector:
                    vz, vy, vx = mgrid[:, 1, j, k] - mgrid[:, 0, j, k]
                    norm = np.sqrt(vx**2 + vy**2 + vz**2)
                    out_vectors.append([vx/norm, vy/norm, vz/norm])
    if outvector:
        return np.array(out_coords), np.array(out_vectors)
    else:
        return np.array(out_coords)


def main_global2local(coords, mgrid, dist=3, leave_boundary=False, outvector=False):
    out_coords = []
    out_vectors = []
    coords = coords - 1 # change to start from 0
    nz,ny,nx = mgrid.shape[1:]
    zz, yy, xx = mgrid.reshape((3, -1))
    z_id, y_id, x_id = np.mgrid[0:nz, 0:ny, 0:nx].reshape((3, -1))
    if outvector:
        vmgrid = mgrid[:,1,:,:] - mgrid[:,0,:,:]
        vmgrid = np.repeat(np.expand_dims(vmgrid,axis=1),mgrid.shape[1],axis=1)
        vzz, vyy, vxx = vmgrid.reshape((3, -1))

    tree = KDTree(np.array([xx, yy, zz]).T, balanced_tree=True)  # balanced_tree=False is faster
    dd, ii = tree.query(coords, k=1, distance_upper_bound=dist+1, workers=-1)
    for i in range(len(coords)):
        d = dd[i]
        idx = ii[i]
        if d <= dist:
            x, y, z = coords[i] + 1 # change to start from 1
            lx, ly, lz = x_id[idx] + 1, y_id[idx] + 1, z_id[idx] + 1
            if not leave_boundary:
                if lx in [1, nx] or ly in [1, ny] or lz in [1, nz]:
                    continue
            out_coords.append([x, y, z, lx, ly, lz])
            if outvector:
                vx, vy, vz = vxx[idx], vyy[idx], vzz[idx]
                norm = np.sqrt(vx**2 + vy**2 + vz**2)
                out_vectors.append([vx/norm, vy/norm, vz/norm])
    if outvector:
        return np.array(out_coords), np.array(out_vectors)
    else:
        return np.array(out_coords)


def main(args):
    mgrid_surf = np.load(args.npy)
    coord_in = np.loadtxt(args.coord, ndmin=2)
    coord_in = coord_in[:, :3]
    if args.outvector is not None:
        if args.invert:
            coords_out, vectors_out = main_global2local(coord_in, mgrid_surf, args.dist, args.leave_boundary, True)
        else:
            coords_out, vectors_out = main_local2global(coord_in, mgrid_surf, True)
    else:
        if args.invert:
            coords_out = main_global2local(coord_in, mgrid_surf, args.dist, args.leave_boundary, False)
        else:
            coords_out = main_local2global(coord_in, mgrid_surf, False)

    if len(coords_out) == 0:
        print("no coords output!")
    else:
        np.savetxt(args.out, coords_out, fmt='%6.1f')
        if args.outvector:
            np.savetxt(args.outvector, vectors_out, fmt='%6.3f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert flatten tomo coords to raw tomo coords. or invert")
    parser.add_argument('--coord', '--i', type=str, required=True,
                        help='coords file name, xyz start from 1')
    parser.add_argument('--npy', '--n', type=str, required=True,
                        help='npy file name')
    parser.add_argument('--out', '--o', type=str, required=True,
                        help='output file name. contain input xyz and converted xyz')
    parser.add_argument('--outvector', '--ov', type=str,
                        help='can output norm vectors in this file, default not.')
    parser.add_argument('--invert', '--inv', action='store_true',
                        help='convert raw coords to flatten coords. skip coords not in flatten tomo')
    parser.add_argument('--leave_boundary', '--lv', action='store_true',
                        help='only used when --invert. default not, becasue result coords on boundary may be inaccurate')
    parser.add_argument('--dist', '--d', type=float, default=3,
                        help='only used in search when --invert. increase it if points are missed(in general unnecessary). in pixel, default 3')
    args = parser.parse_args()
    main(args)
    