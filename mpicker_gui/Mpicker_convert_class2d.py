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

import os,argparse
import numpy as np
from operator import itemgetter
from typing import Dict, List, Tuple
from Mpicker_convert_2dto3d import process_2d, process_3d, get_result
from mpicker_star import read_star_loop


THUID = {"ImgPath": 2, "Quat0": 11, "Quat1": 12, "TX": 15, "TY": 16, "ClassID": 18} # start from 1


def write_out(fout, class2d_data, name_length, is_thu=True, out_star=False):
    if is_thu:
        if out_star:
            head = ("\ndata_\n\nloop_\n_name #1\n_idx #2\n_Quat0 #3\n_Quat1 #4\n_TX #5\n_TY #6\n_ClassID #7\n"
                    "_x #8\n_y #9\n_z #10\n_rot #11\n_tilt #12\n_psi #13\n_dx #14\n_dy #15\n_dz #16\n")
        else:
            head = (f"{'# name':{name_length+1}s} {'idx':>6s} {'Quat0':>10s} {'Quat1':>10s} {'TX':>10s} {'TY':>10s} {'ClassID':>7s} "
                    f"{'x':>8s} {'y':>8s} {'z':>8s} {'rot':>7s} {'tilt':>7s} {'psi':>7s} {'dx':>7s} {'dy':>7s} {'dz':>7s} \n")
        with open(fout, 'w') as f:
            f.write(head)
            for d in class2d_data:
                line = (f"{d[1]:{name_length+1}s} {d[2]:06d} {d[3]:10.6f} {d[4]:10.6f} {d[5]:10.2f} {d[6]:10.2f} {d[7]:7d} "
                        f"{d[8]:8.2f} {d[9]:8.2f} {d[10]:8.2f} {d[11]:7.2f} {d[12]:7.2f} {d[13]:7.2f} {d[14]:7.2f} {d[15]:7.2f} {d[16]:7.2f} \n")
                f.write(line)
    else:
        if out_star:
            head = (f"\ndata_\n\nloop_\n_name #1\n_idx #2\n_rlnAnglePsi #3\n_rlnOriginX #4\n_rlnOriginY #5\n_rlnClassNumber #6\n"
                    f"_x #7\n_y #8\n_z #9\n_rot #10\n_tilt #11\n_psi #12\n_dx #13\n_dy #14\n_dz #15\n")
        else:
            head = (f"{'# name':{name_length+1}s} {'idx':>6s} {'rlnAnglePsi':>11s} {'rlnOriginX':>15} {'rlnOriginY':>15s} {'rlnClassNumber':>14s} "
                    f"{'x':>8s} {'y':>8s} {'z':>8s} {'rot':>7s} {'tilt':>7s} {'psi':>7s} {'dx':>7s} {'dy':>7s} {'dz':>7s} \n")
        with open(fout, 'w') as f:
            f.write(head)
            for d in class2d_data:
                line = (f"{d[1]:{name_length+1}s} {d[2]:06d} {d[3]:11.2f} {d[4]:15.2f} {d[5]:15.2f} {d[6]:14d} "
                        f"{d[7]:8.2f} {d[8]:8.2f} {d[9]:8.2f} {d[10]:7.2f} {d[11]:7.2f} {d[12]:7.2f} {d[13]:7.2f} {d[14]:7.2f} {d[15]:7.2f} \n")
                f.write(line)
    return


def write_out2(fout2, result, name_length, is_thu=True, out_star=False):
    name_class = 'ClassID' if is_thu else 'rlnClassNumber'
    with open(fout2, 'w') as f:
        if out_star:
            head = (f"\ndata_\n\nloop_\n_name #1\n_idx #2\n_rlnCoordinateX #3\n_rlnCoordinateY #4\n_rlnCoordinateZ #5\n"
                    f"_rlnAngleRot #6\n_rlnAngleTilt #7\n_rlnAnglePsi #8\n_rlnOriginX #9\n_rlnOriginY #10\n_rlnOriginZ #11\n_{name_class} #12\n")
        else:
            head = (f"{'# name':{name_length+1}s} {'idx':>6s} {'rlnCoordinateX':>14s} {'rlnCoordinateY':>14s} {'rlnCoordinateZ':>14s} "
                    f"{'rlnAngleRot':>12s} {'rlnAngleTilt':>12s} {'rlnAnglePsi':>12s} {'rlnOriginX':>10s} {'rlnOriginY':>10s} {'rlnOriginZ':>10s} {name_class:>7s} \n")
        f.write(head)
        for d in result:
            line = (f"{d[0]:{name_length+1}s} {d[1]:06d} {d[2]:14.2f} {d[3]:14.2f} {d[4]:14.2f} "
                    f"{d[5]:12.2f} {d[6]:12.2f} {d[7]:12.2f} {d[8]:10.2f} {d[9]:10.2f} {d[10]:10.2f} {d[11]:7d} \n")
            f.write(line)
    return


def main(args):
    fdata = args.data
    fthu = args.thu
    fstar = args.star
    fout = args.out
    pixel = args.pixel
    do_sort = not args.no_sort
    out_star = args.out_star
    fmove = args.fmove
    fout2 = args.out2

    if fthu is None and fstar is None:
        raise Exception("Please provide either thu or star file")
    if fthu is not None and fstar is not None:
        raise Exception("Please provide only one of thu or star file")
    if fthu is not None:
        is_thu = True
    else:
        is_thu = False

    print("read data")
    mrcs2data: Dict[str, Tuple[int, List[List[float]]]] = dict()
    name_length = 10
    with open(fdata, 'r') as f:
        data=f.readlines()
    for i, line in enumerate(data):
        line = line.strip().split()
        if len(line) < 2 or line[0][0] == "#":
            continue
        name_mrcs = os.path.basename(line[0])
        print(name_mrcs)
        name_data = line[1]
        # store order and data of this tomo
        data_proj = np.loadtxt(name_data, ndmin=2)
        if data_proj.shape[1] < 9:
            data_proj = np.insert(data_proj[:, :6], 6, np.zeros((3, 1)), axis=1)
        mrcs2data[name_mrcs] = (i,data_proj[:, 0:9].tolist())
        if len(name_mrcs) > name_length:
            name_length = len(name_mrcs)
    
    if is_thu:
        print("read thu")
        class2d_data = []
        with open(fthu, 'r') as f:
            data = f.readlines()
        for line in data:
            line = line.strip().split()
            if len(line) < 19 or line[0] != "IMG":
                continue
            ImgPath = line[THUID["ImgPath"]] # not THUID - 1 here, because the 1st column is "IMG"
            mrcs = os.path.basename(ImgPath.split("@")[1])
            idx = int(ImgPath.split("@")[0])
            order = mrcs2data[mrcs][0]
            data_thu = [float(line[THUID["Quat0"]]), float(line[THUID["Quat1"]]), float(line[THUID["TX"]]), 
                        float(line[THUID["TY"]]), int(line[THUID["ClassID"]])]
            data_proj = mrcs2data[mrcs][1][idx-1]
            class2d_data.append([order, mrcs, idx] + data_thu + data_proj)

    else:
        print("read star")
        class2d_data = []
        keys, data = read_star_loop(fstar)
        if len(data) == 0:
            keys, data = read_star_loop(fstar, "particles")
        xyangst = "rlnOriginXAngst" in keys and "rlnOriginYAngst" in keys
        if xyangst and pixel is None:
            raise Exception("found rlnOriginXAngst and rlnOriginYAngst, please provide pixel size for conversion")
        for line in data:
            ImgPath = line["rlnImageName"]
            mrcs = os.path.basename(ImgPath.split("@")[1])
            idx = int(ImgPath.split("@")[0])
            ang = float(line["rlnAnglePsi"])
            classid = int(line["rlnClassNumber"])
            if xyangst:
                dx, dy = float(line["rlnOriginXAngst"])/pixel, float(line["rlnOriginYAngst"])/pixel
            else:
                dx, dy = float(line["rlnOriginX"]), float(line["rlnOriginY"])
            order = mrcs2data[mrcs][0]
            data_proj = mrcs2data[mrcs][1][idx-1]
            class2d_data.append([order, mrcs, idx] + [ang, dx, dy, classid] + data_proj)

    if do_sort:
        class2d_data.sort(key=itemgetter(0,2))

    if fout is not None:
        print("write out")
        write_out(fout, class2d_data, name_length, is_thu, out_star)

    if fout2 is None:
        return
    
    print("convert 2dto3d")
    result = []
    if fmove is None:
        data_name = [d[1:3] for d in class2d_data]
        if is_thu:
            data_class2d = np.array([d[3:7] for d in class2d_data])
            data_cls = [d[7] for d in class2d_data]
            data_proj = np.array([d[8:17] for d in class2d_data])
        else:
            data_class2d = np.array([d[3:6] for d in class2d_data])
            data_cls = [d[6] for d in class2d_data]
            data_proj = np.array([d[7:16] for d in class2d_data])
        data_class2d, _ = process_2d(data_class2d, not is_thu, 0, 0, 0)
        data_proj = process_3d(data_proj, False) # assume not norm vector
        data_convert = get_result(data_class2d, data_proj)
        for n, d, c in zip(data_name, data_convert, data_cls):
            line = n + list(d) + [c]
            result.append(line)
    else:
        data_move = np.loadtxt(fmove, ndmin=2)[:, 0:4]
        order = []
        for c, x, y, r in data_move:
            c = int(round(c))
            if is_thu:
                data_name = [d[1:3] for d in class2d_data if d[7]==c]
                data_class2d = np.array([d[3:7] for d in class2d_data if d[7]==c])
                data_proj = np.array([d[8:17] for d in class2d_data if d[7]==c])
                order += [i for i in range(len(class2d_data)) if class2d_data[i][7]==c]
            else:
                data_name = [d[1:3] for d in class2d_data if d[6]==c]
                data_class2d = np.array([d[3:6] for d in class2d_data if d[6]==c])
                data_proj = np.array([d[7:16] for d in class2d_data if d[6]==c])
                order += [i for i in range(len(class2d_data)) if class2d_data[i][6]==c]
            data_class2d, _ = process_2d(data_class2d, not is_thu, x, y, r)
            data_proj = process_3d(data_proj, False) # assume not norm vector
            data_convert = get_result(data_class2d, data_proj)
            for n, d in zip(data_name, data_convert):
                line = n + list(d) + [c]
                result.append(line)
        order = np.argsort(order) # to keep the original order
        result = [result[i] for i in order]

    print("write out2")
    write_out2(fout2, result, name_length, is_thu, out_star)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get information from Class2D result file (.star or .thu) and convert to 3D orientation. \
                                     an advanced version of Mpicker_convert_2dto3d.py")
    parser.add_argument('--data', type=str, required=True,
                        help='file contains 2 column, 1st is name of projection mrcs file, \
                              2nd is the corresponding data used by Mpicker_2dprojection to generate this projection')
    parser.add_argument('--thu', type=str,
                        help='input class2d result like Meta_Final.thu, if use THUNDER2')
    parser.add_argument('--star', type=str,
                        help='input class2d result like run_it025_data.star, if use Relion')
    parser.add_argument('--out', type=str,
                        help='output file contains mrcs_name,index,Quat0,Quat1,TX,TY,class,x,y,z,rot,tilt,psi for each particle')
    parser.add_argument('--pixel', type=float,
                        help='if Relion use rlnOriginX(Y)Angst instead of rlnOriginX(Y) in star file when Class2D, \
                              you need provide pixel size (in A) for conversion')
    parser.add_argument('--no_sort', action='store_true',
                        help='by default, will sort particles in thu by the order of name in data file and particle index')
    parser.add_argument('--out_star', action='store_true',
                        help='write output in star file format. default is simple txt file')
    parser.add_argument('--fmove', type=str,
                        help='file should contain 4 columns, cls,movex,movey,rotate, same as that in Mpicker_convert_2dto3d, \
                              provide this to convert to 3d angles directly (see out2 below). \
                              if not provided, will convert all cls and apply no additional move and rotate')
    parser.add_argument('--out2', type=str,
                        help='output file contains mrcs_name,index,x,y,z,rot,tilt,psi,dx,dy,dz,cls for each particle')
    
    args = parser.parse_args()
    main(args)