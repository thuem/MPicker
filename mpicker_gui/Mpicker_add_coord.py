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
import os
import numpy as np
from Mpicker_particles import ParticleData
from shutil import copyfile


def add_particles(fcoord, fout, fnpy, cls_id, down, clear):
    if cls_id < 1:
        raise Exception("cls_id should >= 1")
    if os.path.isfile(fout):
        copyfile(fout, fout+"~")
        datas = np.loadtxt(fout, ndmin=2)
    else:
        open(fout, mode='w').close()
        datas = np.array([[]])

    coords = np.loadtxt(fcoord, ndmin=2)[:, :3]
    list1 = []
    list2 = []
    list3 = []
    for line in datas:
        if len(line) == 0:
            continue
        particle = ParticleData(line)
        if particle.class_idx < cls_id:
            list1.append(particle.aslist())
        elif particle.class_idx == cls_id:
            if not clear:
                list2.append(particle.aslist())
        else:
            list3.append(particle.aslist())
    mgrid = np.load(fnpy)
    for x, y, z in coords:
        particle = ParticleData(x, y, z, cls_id)
        if down:
            particle.invert_norm()
        list2.append(particle.final_list(mgrid))
    result = np.array(list1+list2+list3)
    np.savetxt(fout, result, fmt=ParticleData.fmt(), header=ParticleData.header())
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="add xyz coords to existed _SelectPoints.txt file")
    parser.add_argument('--coord', '-i', type=str, required=True,
                        help='coords file name, xyz start from 1')
    parser.add_argument('--out', '-o', type=str, required=True,
                        help='existed _SelectPoints.txt file name')
    parser.add_argument('--npy', '-n', type=str, required=True,
                        help='npy file name')
    parser.add_argument('--cls', type=int, default=1,
                        help='class id, by default 1')
    parser.add_argument('--down', action='store_true',
                        help='set UpDown to down, by default up')
    parser.add_argument('--clear', action='store_true',
                        help='clear existed particles with same class id')
    args = parser.parse_args()

    fcoord = args.coord
    fout = args.out
    fnpy = args.npy
    cls_id = args.cls
    down =  args.down
    clear = args.clear

    add_particles(fcoord, fout, fnpy, cls_id, down, clear)
