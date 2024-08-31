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
from pathlib import Path
import subprocess
import numpy as np
import mrcfile
from Mpicker_epicker import make_tmpdir, resolve_z, pad_image, remove_tmpdir, read_thi, filt_score, clear_near_3d
from Mpicker_add_coord import  add_particles
from tqdm import tqdm
from time import time, sleep


def end_replace(s:str, old:str, new:str) -> str:
    if s.endswith(old):
        return s[:-len(old)] + new
    else:
        raise Exception(f"{s} not end with {old}")


def pick_pre(data_path, volume, z_range, pad, pre="", use_t=None, save_t=None):
    with mrcfile.mmap(volume, mode='r') as mrc:
        volume_data = mrc.data

    if use_t is not None: # skip EPicker
        return 0
    
    if save_t is not None: # pick all z slices
        z_range = "all"

    if z_range is None or z_range=="all":
        z_range = np.arange(1, len(volume_data) + 1) # all slices
    else:
        z_range = resolve_z(z_range)
    prefix = pre + Path(volume).name.replace('.mrc', '')
    num = 0
    for z in z_range:
        # convert to images
        if z > len(volume_data):
            continue
        out_name = str( Path(data_path)/f'{prefix}_z{z:02d}.mrc' )
        img_data = volume_data[z - 1]
        if pad > 0:
            img_data = pad_image(img_data, pad)
        with mrcfile.new(out_name, overwrite=True) as mrc:
            mrc.set_data(img_data.astype(np.float32))
        num += 1
    return num


def run_epicker(cmd, total, result_dir):
    result_dir = Path(result_dir)
    process = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.DEVNULL)
    # cannot get process.stdout in real time, because output buffer?
    num = 0
    pbar = tqdm(total=total)
    while process.poll() is None:
        num2 = len(list(result_dir.glob("*.thi")))
        if num2 > num and num2 <= total:
            pbar.set_description("", refresh=False)
            pbar.update(num2-num)
            num = num2
        elif num2 == 0:
            pbar.set_description("loading", refresh=False)
            pbar.update(0)
        sleep(1)
    num2 = min(total, len(list(result_dir.glob("*.thi"))))
    pbar.update(num2 - num)
    pbar.close()
    process.wait()
    return process.returncode


def pick_post(result_path, pick_out, volume, z_range, edgex, edgey, thres, max_num, sigma, dist, pre="", 
              use_t=None, save_t=None):
    with mrcfile.mmap(volume, mode='r') as mrc:
        volume_data = mrc.data
    if z_range is None or z_range=="all":
        z_range = np.arange(1, len(volume_data) + 1) # all slices
    else:
        z_range = resolve_z(z_range)
    prefix = pre + Path(volume).name.replace('.mrc', '')

    if use_t is not None:
        xyz_score = np.loadtxt(use_t, ndmin=2)
        if save_t is not None:
            np.savetxt(save_t, xyz_score) # just copy it
        if len(xyz_score) > 0:
            mask = np.isin(xyz_score[:, 2], z_range)
            xyz_score = xyz_score[mask]
    elif use_t is None and save_t is not None:
        xyz_score = []
        for z in np.arange(1, len(volume_data) + 1):
            out_name = str( Path(result_path)/f'{prefix}_z{z:02d}.thi' )
            xyz_s = read_thi(out_name, z)
            xyz_score += xyz_s
        xyz_score = np.array(xyz_score)
        np.savetxt(save_t, xyz_score)
        if len(xyz_score) > 0:
            mask = np.isin(xyz_score[:, 2], z_range)
            xyz_score = xyz_score[mask]
    elif use_t is None and save_t is None:
        xyz_score = []
        for z in z_range: 
            # convert xy to xyz
            out_name = str( Path(result_path)/f'{prefix}_z{z:02d}.thi' )
            xyz_s = read_thi(out_name, z)
            xyz_score += xyz_s
        xyz_score = np.array(xyz_score)

    if len(xyz_score) > 0:
        # apply edge and thres
        mask = (xyz_score[:, 0] > edgex) & (xyz_score[:, 0] < volume_data.shape[2] - edgex) \
             & (xyz_score[:, 1] > edgey) & (xyz_score[:, 1] < volume_data.shape[1] - edgey) \
             & (xyz_score[:, 3] > thres)
        xyz_score = xyz_score[mask]

    if len(xyz_score) > 0 and max_num > 0:
        # apply gaussian filter in 3d
        if sigma > 0.25:
            score = filt_score(xyz_score, sigma)[:, 3]
        else:
            score = xyz_score[:, 3]
        # apply dist and max_num
        xyz_score = xyz_score[np.argsort(score)[::-1]] # score big to small
        if dist >= 0:
            mask = clear_near_3d(xyz_score, dist)
            xyz_score = xyz_score[mask]
        if len(xyz_score) > max_num:
            xyz_score = xyz_score[:max_num]
        np.savetxt(pick_out, xyz_score, fmt='%4d %4d %4d %.3f') # score after filter
    else:
        np.savetxt(pick_out, np.array([]))
    return


def read_vzd(fin):
    volume_list = []
    zrange_list = []
    down_list = []
    with open(fin, mode='r') as f:
        data = f.readlines()
    for line in data:
        line = line.strip()
        if len(line)==0 or line[0] == "#":
            continue
        line = line.split()
        if len(line) >= 2:
            volume_list.append(line[0])
            zrange_list.append(line[1])
        if len(line) >= 3:
            down_list.append(line[2]=="-1")
        else:
            down_list.append(False) # up by default
    return volume_list, zrange_list, down_list


def check_tmp(use_tmp, save_tmp, overwrite_tmp, volume_list, max_num):
    if use_tmp is not None:
        use_tmp_list = []
        suffix = f"_epickerTmp{max_num}_id{use_tmp}.txt"  # suffix of tmp file in MPicker
        for volume in volume_list:
            ftmp = end_replace(volume, "_result.mrc", suffix)
            if Path(ftmp).is_file():
                use_tmp_list.append(ftmp)
            else:
                use_tmp_list.append(None)
    else:
        use_tmp_list = [None] * len(volume_list)

    if save_tmp is not None:
        save_tmp_list = []
        suffix = f"_epickerTmp{max_num}_id{save_tmp}.txt"
        for volume in volume_list:
            ftmp = end_replace(volume, "_result.mrc", suffix)
            if Path(ftmp).is_file() and not overwrite_tmp:
                save_tmp_list.append(None)
            else:
                save_tmp_list.append(ftmp)
    else:
        save_tmp_list = [None] * len(volume_list)

    return use_tmp_list, save_tmp_list


def main(model, fin, dirout, add_cls, overwrite, thres, max_num, 
         sigma=0, dist=-1, edgex=0, edgey=0, pad=-1, gpuid=0, use_tmp=None, save_tmp=None, overwrite_tmp=True, 
         save_epicker_out=False, epicker_path='epicker.sh'):
    print("preprocessing...")
    volume_list, zrange_list, down_list = read_vzd(fin)
    tmp_path = make_tmpdir(Path(dirout))
    data_path = Path(tmp_path)/'data'
    result_path = Path(tmp_path)/'result'
    data_path.mkdir()
    result_path.mkdir()

    if add_cls > 0 or use_tmp is not None or save_tmp is not None:
        for volume in volume_list:
            if not volume.endswith("_result.mrc"):
                print(volume)
                raise Exception("add_cls or use_tmp or save_tmp require mrc file end with _result.mrc")

    use_tmp_list, save_tmp_list = check_tmp(use_tmp, save_tmp, overwrite_tmp, volume_list, max_num)
    
    total_num = 0
    for i, (volume, zrange, use_t, save_t) in tqdm(enumerate(zip(volume_list, zrange_list, use_tmp_list, save_tmp_list)), 
                                    total=len(volume_list)):
        num = pick_pre(str(data_path), volume, zrange, pad, f"{i}_", use_t, save_t)
        total_num += num

    if total_num == 0:
        print("just use tmp file, skip EPicker")
    else:
        print("picking...")
        if model is None:
            raise Exception("EPicker require model file, provide it by --model")
        cmd = f'{epicker_path} --data {str(data_path)} --load_model {model} --K {max_num} --output {str(result_path)} --edge 0 --gpus {gpuid}'
        print(cmd)
        s = run_epicker(cmd, total_num, result_path)
        if s != 0:
            if not save_epicker_out:
                remove_tmpdir(tmp_path)
            print("exit code", s)
            raise Exception("epicker failed, see terminal for detail")
    
    print("postprocessing...")
    for i, (volume, zrange, down, use_t, save_t) in tqdm(enumerate(zip(volume_list, zrange_list, down_list, 
                                                                       use_tmp_list, save_tmp_list)), total=len(volume_list)):
        pick_out = Path(dirout)/(Path(volume).stem+f"_epickerCoord_{i}.txt")
        # while pick_out.is_file():
        #     pick_out = Path(str(pick_out)+".txt")  # may have same name...
        pick_post(str(result_path), str(pick_out), volume, zrange, edgex, edgey, thres, max_num, sigma, dist, f"{i}_", 
                  use_t, save_t)
        if add_cls > 0:
            store_file = end_replace(volume, "_result.mrc", "_SelectPoints.txt")
            npy_file = end_replace(volume, "_result.mrc", "_convert_coord.npy")
            add_particles(str(pick_out), store_file, npy_file, add_cls, down, overwrite)
    if save_epicker_out:
        print("epicker_out", tmp_path)
    else:
        remove_tmpdir(tmp_path)


def main_fromtmp(use_epicker_out, fin, dirout, add_cls, overwrite, thres, max_num, sigma=0, dist=-1, edgex=0, edgey=0,
                 use_tmp=None, save_tmp=None, overwrite_tmp=True):
    volume_list, zrange_list, down_list = read_vzd(fin)
    result_path = Path(use_epicker_out)/'result'
    Path(dirout).mkdir(parents=True, exist_ok=True)

    if add_cls > 0 or use_tmp is not None or save_tmp is not None:
        for volume in volume_list:
            if not volume.endswith("_result.mrc"):
                print(volume)
                raise Exception("add_cls or use_tmp or save_tmp require mrc file end with _result.mrc")

    use_tmp_list, save_tmp_list = check_tmp(use_tmp, save_tmp, overwrite_tmp, volume_list, max_num)

    print("postprocessing...")
    for i, (volume, zrange, down, use_t, save_t) in tqdm(enumerate(zip(volume_list, zrange_list, down_list, 
                                                                       use_tmp_list, save_tmp_list)), total=len(volume_list)):
        pick_out = Path(dirout)/(Path(volume).stem+f"_epickerCoord_{i}.txt")
        # while pick_out.is_file():
        #     pick_out = Path(str(pick_out)+".txt")  # may have same name...
        pick_post(str(result_path), str(pick_out), volume, zrange, edgex, edgey, thres, max_num, sigma, dist, f"{i}_", 
                  use_t, save_t)
        if add_cls > 0:
            store_file = end_replace(volume, "_result.mrc", "_SelectPoints.txt")
            npy_file = end_replace(volume, "_result.mrc", "_convert_coord.npy")
            add_particles(str(pick_out), store_file, npy_file, add_cls, down, overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,
                        help='the path of model used when pick')
    parser.add_argument('--fin', type=str, required=True,
                        help='the txt file that records mrc file name and its z range in each line, shuch as "xxx.mrc 12-14". ' + \
                            'the 3rd column (optional) can be 1 or -1 to specify up or down')
    parser.add_argument('--out', type=str, required=True,
                        help='the folder to save pick results, xyz start from 1 and score')
    parser.add_argument('--add_cls', type=int, default=0,
                        help='if >0, save the results into _SelectPoints.txt for each _result.mrc with this class id')
    parser.add_argument('--overwrite', action='store_true',
                        help='clear particles in _SelectPoints.txt with same class id')
    parser.add_argument('--thres', type=float, default = 0.1,
                        help='minimum score to save particle, 0 to 1')
    parser.add_argument('--max_num', type=int, default = 500,
                        help='maximum number to save particle')
    parser.add_argument('--dist', type=float, default = -1,
                        help='minimum distance(3d) between particles when pick, in pixel')
    parser.add_argument('--sigma', type=float, default = 2,
                        help='apply 3d gauss filter to epicker result, after thres, before dist, in pixel')
    parser.add_argument('--edgex', type=int, default = 10,
                        help='ignore the particle near edge(x) when pick, in pixel')
    parser.add_argument('--edgey', type=int, default = 10,
                        help='ignore the particle near edge(y) when pick, in pixel')
    parser.add_argument('--pad', type=int, default = -1,
                        help='pad to square with this size when pick and generate label, useful for old epicker, in pixel')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu to use when pick or train')
    
    parser.add_argument('--use_tmp', type=str, default=None,
                        help='will use tmp file (as in GUI) with the provided output id if exist, so can just run post process')
    parser.add_argument('--save_tmp', type=str, default=None,
                        help='you can provide an output id here to generate tmp file (as in GUI) for each flattened tomo when first run. slower.')
    parser.add_argument('--not_overwrite_tmp', action='store_true',
                        help='not overwrite the tmp file if exist. only useful when save_tmp is provided')
    
    parser.add_argument('--save_epicker_out', action='store_true',
                        help='not remove the tmp folder for EPicker after pick')
    parser.add_argument('--use_epicker_out', type=str, default=None,
                        help='provide an existed EPicker tmp folder and just run post process')
    parser.add_argument('--epicker_path', type=str, default='epicker.sh',
                        help='you can specify the path of epicker script explicitly')
    
    args = parser.parse_args()

    t0 = time()
    overwrite_tmp = not args.not_overwrite_tmp
    if args.use_epicker_out is None:
        if args.use_tmp is None and args.model is None:
            raise Exception("require model file, provide it by --model")
        
        if args.use_tmp is not None and args.save_tmp is not None and overwrite_tmp:
            print("both use_tmp and save_tmp and overwrite_tmp, so will use_tmp (not recalculate) if tmp file exist.")
            
        main(args.model, args.fin, args.out, args.add_cls, args.overwrite, args.thres, args.max_num, 
            args.sigma, args.dist, args.edgex, args.edgey, args.pad, args.gpuid, args.use_tmp, args.save_tmp, overwrite_tmp, 
            args.save_epicker_out, args.epicker_path)
    else:
        print("Notice: you use existed EPicker output folder:", args.use_epicker_out)
        print("if you not set use_tmp, make sure the existed folder did not set use_tmp too")
        print("if you set save_tmp, make sure the existed folder setted save_tmp too")
        print("otherwise, the result may be incomplete")

        main_fromtmp(args.use_epicker_out, args.fin, args.out, args.add_cls, args.overwrite, args.thres, args.max_num, 
                    args.sigma, args.dist, args.edgex, args.edgey, args.use_tmp, args.save_tmp, overwrite_tmp)
        
    print("finish", time()-t0)