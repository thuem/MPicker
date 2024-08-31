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

import sys 
import numpy as np
import mrcfile
from mpicker_core import interp_mgrid2tomo

if len(sys.argv) not in [3, 4]:
    print("useage:")
    print(sys.argv[0], "input_mrc input_npy output_mrc")
    print("or", sys.argv[0], "input_mrc in_out_txt")
    quit()

def convert_npymrc(tomo, input_npy, output, voxel_size=1):
    mgrid_surf=np.load(input_npy)
    flatten_tomo = interp_mgrid2tomo(mgrid_surf, tomo)
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(flatten_tomo.astype(np.float32))
        mrc.voxel_size=voxel_size

input_mrc=sys.argv[1]
try:
    with mrcfile.open(input_mrc) as mrc:
        tomo=mrc.data
        voxel_size=mrc.voxel_size
except:
    with mrcfile.mmap(input_mrc) as mrc:
        tomo=mrc.data
        voxel_size=mrc.voxel_size
    print("memory not enough, might be slow")

if len(sys.argv) == 4:
    input_npy = sys.argv[2]
    output = sys.argv[3]
    convert_npymrc(tomo, input_npy, output, voxel_size)

elif len(sys.argv) == 3:
    input_txt = sys.argv[2]
    with open(input_txt, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        if len(line) < 2 or line[0] == "#":
            continue
        input_npy, output = line[:2]
        print(input_npy, output)
        convert_npymrc(tomo, input_npy, output, voxel_size)
