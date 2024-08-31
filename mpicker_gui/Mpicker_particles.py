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

import argparse, warnings
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from math import cos, sin, tan


class ParticleData(object):
    def __init__(self, *arg):
        """
        input can be:\n
        3 int: x y z \n
        4 int: x y z class_idx \n
        1 List[float] with length 16 \n
        1 List[float] with length 17 (17th is class_idx)
        """
        if len(arg) in [3, 4]:
            # input only x y z (or class_idx)
            x,  y,  z = arg[0:3]
            # coord in flatten tomo, start from 1, int
            self.x = x 
            self.y = y
            self.z = z
            # second point (optional)
            self.x2 = -1 
            self.y2 = -1
            self.z2 = -1
            # coord in raw tomo, start from 1, float
            self.Rx = -1 
            self.Ry = -1
            self.Rz = -1
            self.Rx2 = -1 
            self.Ry2 = -1
            self.Rz2 = -1
            # norm vector, nxnynz correspond to up
            self.up = 1 # Up:1  Down:-1
            self.nx = 0
            self.ny = 0
            self.nz = 0
            # class index, start from 1
            if len(arg)>3:
                self.class_idx = arg[3]
            else:
                self.class_idx = 1
        elif len(arg)==1:
            # input one list (or numpy array)
            self.x,  self.y,  self.z,  self.x2,  self.y2,  self.z2, \
            self.Rx, self.Ry, self.Rz, self.Rx2, self.Ry2, self.Rz2, self.up, \
            self.nx, self.ny, self.nz = arg[0][0:16]
            if len(arg[0]) > 16:
                self.class_idx = arg[0][16]
            else:
                self.class_idx = 1

        self.x,  self.y,  self.z = int(round(self.x)),  int(round(self.y)), int(round(self.z))
        self.x2,  self.y2,  self.z2 = int(round(self.x2)),  int(round(self.y2)),  int(round(self.z2))
        self.class_idx = int(self.class_idx)

    def aslist(self):
        data = [self.x,  self.y,  self.z,  self.x2,  self.y2,  self.z2, \
                self.Rx, self.Ry, self.Rz, self.Rx2, self.Ry2, self.Rz2, self.up, \
                self.nx, self.ny, self.nz, self.class_idx]
        return data

    @staticmethod
    def header():
        # the order to save
        header = ("%5s"+"\t%7s"*16)%\
                    ('x','y','z','x2','y2','z2','Raw_x','Raw_y','Raw_z', \
                    'Raw_x2','Raw_y2','Raw_z2','UpDown','nx','ny','nz','class')
        header = header + "  # xyz start from 1"
        return header

    @staticmethod
    def fmt():
        # for np.savetxt
        fmt = '\t'.join(['%7d','%7d','%7d','%7d','%7d','%7d','%7.1f','%7.1f','%7.1f', \
                         '%7.1f','%7.1f','%7.1f','%7d','%7.3f','%7.3f','%7.3f','%7d'])
        return fmt   

    def __eq__(self, other):
        if self.x==other.x and self.y==other.y and self.z==other.z and self.class_idx==other.class_idx:
            return True
        else:
            return False

    def set_class(self, idx):
        self.class_idx = int(idx)

    def add_point2(self, x, y, z):
        self.x2 = int(round(x)) 
        self.y2 = int(round(y))
        self.z2 = int(round(z))

    def del_point2(self):
        self.x2 = -1 
        self.y2 = -1
        self.z2 = -1
        self.Rx2 = -1 
        self.Ry2 = -1
        self.Rz2 = -1

    def has_point2(self):
        if (self.x2, self.y2, self.z2) == (-1, -1, -1):
            return False
        else:
            return True

    def invert_norm(self):
        self.up *= -1

    def flipz(self, sx, sz):
        self.x = sx - self.x + 1
        self.z = sz - self.z + 1
        if self.has_point2():
            self.x2 = sx - self.x2 + 1
            self.z2 = sz - self.z2 + 1
        self.invert_norm()

    def UpDown(self, LR=False):
        # up is right, down is left
        if self.up > 0:
            if LR:
                return "R"
            else:
                return "Up"
        elif self.up < 0:
            if LR:
                return "L"
            else:
                return "Down"

    def calculate(self, mgrid):
        sz, sy, sx = mgrid.shape[1:]
        if self.x>=1 and self.x<=sx and self.y>=1 and self.y<=sy and self.z>=1 and self.z<=sz:
            if (self.Rx, self.Ry, self.Rz) == (-1, -1, -1):
                self.Rz, self.Ry, self.Rx = mgrid[:, int(self.z)-1, int(self.y)-1, int(self.x)-1] + 1

            if self.x2>=1 and self.x2<=sx and self.y2>=1 and self.y2<=sy and self.z2>=1 and self.z2<=sz \
                and (self.Rx2, self.Ry2, self.Rz2) == (-1, -1, -1):
                self.Rz2, self.Ry2, self.Rx2 = mgrid[:, int(self.z2)-1, int(self.y2)-1, int(self.x2)-1] + 1

            if (self.nx, self.ny, self.nz) == (0, 0, 0) and sz>1:
                vector = mgrid[:, 1, int(self.y)-1, int(self.x)-1] - mgrid[:, 0, int(self.y)-1, int(self.x)-1]
                self.nz, self.ny, self.nx = vector / np.linalg.norm(vector)
    
    def clear_calculate(self): 
        self.Rx, self.Ry, self.Rz, self.Rx2, self.Ry2, self.Rz2 = -1, -1, -1, -1, -1, -1
        self.nx, self.ny, self.nz = 0, 0, 0

    def final_list(self, mgrid):
        self.calculate(mgrid)
        return self.aslist()

    def __getitem__(self, idx):
        return self.aslist()[idx]




def merge_files(in_path, out_path, dist=-1, exclude_same=False, cls=None):
    particles = []
    scores = []
    idxs = []
    files = [str(p.absolute()) for p in Path(in_path).glob("surface_*/*_SelectPoints.txt") if p.is_file()] + \
            [str(p.absolute()) for p in Path(in_path).glob("manual_*/*_SelectPoints.txt") if p.is_file()]
    if cls is not None and cls < 1:
        print("cls should >= 1, ignore it")
        cls = None

    warnings.filterwarnings("ignore")
    idx = 0
    for file in files:
        try:
            p = []
            s = []
            i = []
            data = np.loadtxt(file, ndmin=2)
            if dist>=0:
                z_mid = Path(file).name.split('_')[5] # thick, eg. surface_45-1_RBF_5_thick_30_SelectPoints.txt
                z_mid = int(z_mid) + 1
            for line in data:
                par=ParticleData(line)
                if cls is not None and par.class_idx != cls:
                    continue
                p.append(par)
                if dist >= 0:
                    score = abs(par.z - z_mid)
                    s.append(score)
                    i.append(idx)
                    idx += 1
            if len(p) > 0:
                particles += p
                scores += s
                idxs += i
                print(Path(file).name, "success")
        except Exception as e:
            print(Path(file).name, "fail", e)        
    warnings.filterwarnings("default")

    if len(particles) == 0:
        print("No particles!")
        return

    result_array = np.array([par.aslist() for par in particles])
    if dist >= 0:
        coords = np.array([ [par.Rx, par.Ry, par.Rz] for par in particles ])
        class_idxs = np.array([par.class_idx for par in particles])
        scores = np.array(scores)
        idxs = np.array(idxs)
        sort_idx = np.argsort(scores) #small to big
        coords = coords[sort_idx]
        class_idxs= class_idxs[sort_idx]

        if (class_idxs == class_idxs[0]).all() or exclude_same == False:
            # has only one particle class, or exclude particle in all class
            only_same_class = False 
        else:
            # only exclude particles in same class
            only_same_class = True

        tree = KDTree(coords)
        pick_idx = np.ones(len(coords), dtype=bool)
        for i in range(len(coords)): # the ith coord
            if pick_idx[i]:
                near_idx=tree.query_ball_point(coords[i], dist) # <=dist are near
                if only_same_class:
                    near_idx = [n for n in near_idx if class_idxs[n]==class_idxs[i]]
                    # near_idx = near_idx[ class_idxs[near_idx]==class_idxs[i] ]
                pick_idx[near_idx] = False
                pick_idx[i] = True # dist from itself is always 0
        
        sort_back_idx = np.argsort(idxs[sort_idx][pick_idx])
        result_array = result_array[sort_idx][pick_idx]
        result_array = result_array[sort_back_idx]

    np.savetxt(out_path, result_array, fmt=ParticleData.fmt(), header=ParticleData.header())
    return


def calculate_angles(in_path, out_path, fill_rot, scale, skip_point2, out_star):
    data = np.loadtxt(in_path, ndmin=2)
    if data.size == 0:
        print("No particles!")
        return
    particles = [ ParticleData(line) for line in data ]
    X = np.array([ par.Rx for par in particles ]) * scale
    Y = np.array([ par.Ry for par in particles ]) * scale
    Z = np.array([ par.Rz for par in particles ]) * scale
    up = np.array([ par.up for par in particles ])
    Nx = np.array([ par.nx for par in particles ]) * up
    Ny = np.array([ par.ny for par in particles ]) * up
    Nz = np.array([ par.nz for par in particles ]) * up
    Ux = np.array([ par.Rx2 - par.Rx for par in particles ])
    Uy = np.array([ par.Ry2 - par.Ry for par in particles ])
    Uz = np.array([ par.Rz2 - par.Rz for par in particles ])
    class_idx = np.array([ par.class_idx for par in particles ])
    if skip_point2:
        skip_rot = [True] * len(particles)
    else:
        skip_rot = [ par.has_point2()==False for par in particles ] # no point2

    norm_n = np.sqrt(Nx**2+Ny**2+Nz**2)
    Nx = Nx/norm_n
    Ny = Ny/norm_n
    Nz = Nz/norm_n

    dot_uv = Nx*Ux + Ny*Uy + Nz*Uz
    Ux = Ux - dot_uv*Nx
    Uy = Uy - dot_uv*Ny
    Uz = Uz - dot_uv*Nz

    norm_u = np.sqrt(Ux**2+Uy**2+Uz**2)
    Ux = Ux/norm_u
    Uy = Uy/norm_u
    Uz = Uz/norm_u

    tilt = np.arccos(Nz)
    psi = np.arctan2(Ny, -Nx) # notice the positive direction of all angles in relion are clockwise

    if fill_rot is None:
        rot = np.random.uniform(-180, 180, len(tilt))
    else:
        rot = np.ones_like(tilt) * fill_rot

    for i,(ux,uy,uz,t,p) in enumerate(zip(Ux,Uy,Uz,tilt,psi)):
        if skip_rot[i]:
            continue
        if sin(t) == 0: # nx=ny=0
            rot[i] = np.arctan2(-uy, ux) * 180/np.pi
        elif abs(tan(p)) <= 1: # |nx|>=|ny|
            rot[i] = np.arctan2( -( sin(p)*cos(t)*uz + sin(t)*uy ) / cos(p), uz ) * 180/np.pi
        else:
            rot[i] = np.arctan2( ( cos(p)*cos(t)*uz - sin(t)*ux ) / sin(p), uz ) * 180/np.pi

    tilt = tilt * 180/np.pi
    psi = psi * 180/np.pi

    result=np.array([X,Y,Z,rot,tilt,psi,class_idx]).T

    if out_star:
        header = ("\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
                  "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n_rlnClassNumber #7")
        np.savetxt(out_path, result,fmt='%7.1f\t'*6+'%7d', header=header, comments='')
    else:
        header =  "%5s\t%7s\t%7s\t%7s\t%7s\t%7s\t%7s"%('X','Y','Z','rot','tilt','psi','class') + "  # xyz start from 1"
        np.savetxt(out_path, result,fmt='%7.1f\t'*6+'%7d', header=header)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="postprocess of _SelectPoints.txt")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['merge', 'angle', 'all'],
                        help='task you want to do. default merge all _SelectPoints.txt and then calculate euler angle')
    parser.add_argument('--input', type=str, 
                        help='mpicker output path of this tomo if merge. or input file name to convert if angle')
    parser.add_argument('--out_mer', type=str, default='SelectPoints_merge.txt',
                        help='output file name of merge file')
    parser.add_argument('--dist', type=float, default=-1,
                        help='will make sure distance (in pixel) between particles > this value after merge. <0 to skip (default)')
    parser.add_argument('--cls', type=int, default=None,
                        help='specify which class id to merge, by default all')
    parser.add_argument('--exclude_all', action='store_true',
                        help='exclude near particles (if dist >=0) even from different class, default only in same class')
    parser.add_argument('--out_ang', type=str, default='SelectPoints_angle.txt',
                        help='output file name of euler angle')
    parser.add_argument('--fill_rot', type=float, default=None,
                        help='the value to fill rot if no point2, [-180,180), default use random')
    parser.add_argument('--scale', type=float, default=1,
                        help='rescale the coordinate in out_ang, default 1. for example, give 1.5 when convert bin3 to bin2')
    parser.add_argument('--skip_point2', action='store_true',
                        help='will fill_rot even if has point2')
    parser.add_argument('--out_star', action='store_true',
                        help='write out_ang in star file format. default is simple txt file')
    args = parser.parse_args()

    if args.mode == 'merge':
        merge_files(args.input, args.out_mer, args.dist, not args.exclude_all, args.cls)
    elif args.mode == 'angle':
        calculate_angles(args.input, args.out_ang, args.fill_rot, args.scale, args.skip_point2, args.out_star)
    elif args.mode == 'all':
        merge_files(args.input, args.out_mer, args.dist, not args.exclude_all, args.cls)
        calculate_angles(args.out_mer, args.out_ang, args.fill_rot, args.scale, args.skip_point2, args.out_star)