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
import sys, os
from shutil import rmtree
import numpy as np
import mrcfile
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter


delete_tmp = True


def absolute_path(path):
    # path = Path(path) # str or Path
    if path is None:
        return None
    else:
        return os.path.abspath(str(path))


def make_tmpdir(path):
    path = Path(path) # parent path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        print("makedirs:", str(path))
    i = 0
    while True:
        name = "tmp_epicker_" + str(i)
        tmp = path/name
        if not tmp.exists():
            break
        i += 1
    tmp.mkdir()
    return absolute_path(tmp)


def remove_tmpdir(path):
    path = Path(path)
    if path.is_dir():
        path = absolute_path(path)
        try:
            rmtree(path)
        except Exception as e:
            print(e)
            print("failed to remove", path)


def resolve_z(zstring):
    # such as 9,12-14
    z_list = []
    for zs in zstring.split(','):
        zz = zs.split('-')
        try:
            if len(zz) == 1:
                z = int(zz[0])
                if z <= 0:
                    raise Exception()
                z_list.append(z)
            elif len(zz) == 2:
                z1, z2 = zz
                z1, z2 = min(int(z1), int(z2)), max(int(z1), int(z2))
                if z1 <= 0 or z2 <= 0:
                    raise Exception()
                z_list += range(z1, z2 + 1)
            else:
                raise Exception()
        except:
            print("check the input", zs)
    return np.array(list(set(z_list)), dtype=int)


def pad_image(image, size):
    s01, s02 = image.shape
    s = max(s01, s02)
    if s > size:
        print("image large than pad size, not recommend. so pad to", s)
        size = s
        # s1 = int(size / s * s01)
        # s2 = int(size / s * s02) 
        # fft = np.fft.rfft2(image)
        # fft1 = fft[0:s1//2, 0:s2//2+1]
        # fft2 = fft[-s1//2:, 0:s2//2+1]
        # fft = np.concatenate([fft1, fft2], axis=0)
        # fft *= (s1*s2)/(s01*s02)
        # image = np.fft.irfft2(fft, s=(s1, s2))

    # s1, s2 = image.shape
    result = np.random.normal(image.mean(), image.std(), (size, size))
    result[:s01, :s02] = image
    return result


def read_thi(thi_path, z):
    xyz_score = []
    if Path(thi_path).is_file():
        with open(thi_path, mode='r') as f:
            data = f.readlines()
        for line in data:
            try:
                line = line.strip().split()
                x = int(line[0]) + 1
                y = int(line[1]) + 1
                score = float(line[2])
                xyz_score.append([x, y, z, score])
            except:
                pass
    else:
        print("Warning: thi file not found,", thi_path)
    return xyz_score


def write_thi(thi_path, coordxyz, z):
    lines = ['[Particle Coordinates: X Y Value]\n']
    coordz = coordxyz[:, 2].astype(int)
    coordxyz = coordxyz[coordz==z][:, :3]
    for x,y,_ in coordxyz:
        lines.append(f'{int(x) - 1} {int(y) - 1}\n')
    lines.append('[End]')
    with open(thi_path, mode='w') as f:
        f.writelines(lines)


def filt_score(xyz_score, sigma):
    xyz_score = xyz_score.copy()
    x, y, z = xyz_score[:, :3].T.astype(int)
    s = xyz_score[:, 3]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    tomo = np.zeros((x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1), dtype=float)
    tomo[x - x_min, y - y_min, z - z_min] = s
    tomo = gaussian_filter(tomo, sigma=sigma, mode='constant')
    s_new = tomo[x - x_min, y - y_min, z - z_min] * (np.sqrt(2*np.pi) * sigma)**3
    xyz_score[:, 3] = s_new
    return xyz_score


def clear_near_3d(xyz_s, dist):
    # assume xyz has been sorted, prefer first
    coords = xyz_s[:, :3]
    tree = KDTree(coords)
    pick_idx = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)): # the ith coord
        if pick_idx[i]:
            near_idx = tree.query_ball_point(coords[i], dist)
            pick_idx[near_idx] = False
            pick_idx[i] = True # dist from itself is always 0
    return pick_idx


def pick_epicker(model, volume, z_range, pick_out, max_num, pad=-1, gpuid='0', \
                 epicker_path='epicker.sh', full_result=False):
    tmp_path = make_tmpdir(Path(pick_out).parent)
    data_path = Path(tmp_path)/'data'
    result_path = Path(tmp_path)/'result'
    data_path.mkdir()
    result_path.mkdir()

    with mrcfile.open(volume) as mrc:
        volume_data = mrc.data
    if z_range is None or full_result:
        z_range = np.arange(1, len(volume_data) + 1)
    else:
        z_range = resolve_z(z_range)
    prefix = Path(volume).name.replace('.mrc', '')

    for z in z_range:
        # convert to images
        if z > len(volume_data):
            continue
        out_name = str( data_path/f'{prefix}_z{z:02d}.mrc' )
        img_data = volume_data[z - 1]
        if pad > 0:
            img_data = pad_image(img_data, pad)
        with mrcfile.new(out_name, overwrite=True) as mrc:
            mrc.set_data(img_data.astype(np.float32))

    cmd = f'{epicker_path} --data {str(data_path)} --load_model {model} --K {max_num} --output {str(result_path)} --edge 0 --gpus {gpuid}'
    print(cmd)
    s = os.system(cmd)
    if s != 0:
        if delete_tmp:
            remove_tmpdir(tmp_path)
        print("exit code", s)
        raise Exception("epicker failed, see terminal for detail")
    
    xyz_score = []
    for z in z_range: 
        # convert xy to xyz
        out_name = str( result_path/f'{prefix}_z{z:02d}.thi' )
        xyz_s = read_thi(out_name, z)
        xyz_score += xyz_s

    if delete_tmp:
        remove_tmpdir(tmp_path)

    return np.array(xyz_score)
    

def pick_main(model, volume, z_range, pick_out, thres, max_num, sigma=0, \
              dist=-1, edgex=0, edgey=0, pad=-1, gpuid='0', epicker_path='epicker.sh',
              tmp_in=None, tmp_out=None):
    with mrcfile.mmap(volume) as mrc:
        volume_shape = mrc.data.shape

    if tmp_in is None and tmp_out is None:
        xyz_score = pick_epicker(model, volume, z_range, pick_out, max_num, pad, gpuid, epicker_path, False)
    else:
        if tmp_in is not None:
            print("use tmp file to pick")
            xyz_score = np.loadtxt(tmp_in, ndmin=2)
        elif tmp_out is not None:
            xyz_score = pick_epicker(model, volume, z_range, pick_out, max_num, pad, gpuid, epicker_path, True)
            print("save tmp file")
            np.savetxt(tmp_out, xyz_score)
        if z_range is not None and xyz_score.size > 0:
            z_range = resolve_z(z_range)
            xyz_score = xyz_score[np.isin(xyz_score[:, 2], z_range)]

    if xyz_score.size > 0:
        # apply edge and thres
        mask = (xyz_score[:, 0] > edgex) & (xyz_score[:, 0] < volume_shape[2] - edgex) \
             & (xyz_score[:, 1] > edgey) & (xyz_score[:, 1] < volume_shape[1] - edgey) \
             & (xyz_score[:, 3] > thres)
        xyz_score = xyz_score[mask]

    if xyz_score.size > 0 and max_num > 0:
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


def label_main(volume, z_range, label_coord, label_dir, pad=-1, filt_col=None, filt_range=None, pre=None):
    label_dir = Path(label_dir)
    label_dir.mkdir(exist_ok=True)
    coords = np.loadtxt(label_coord, ndmin=2) # xyz from 1
    if coords.shape[1] < 3:
        raise Exception("empty label_coord")
    
    if filt_col is not None and filt_range is not None and coords.shape[1] >= filt_col:
        filt_range = resolve_z(filt_range)
        filt = coords[:, filt_col - 1].astype(int)
        coords = coords[np.isin(filt, filt_range)]
        if len(coords) == 0:
            raise Exception("empty label_coord after filt")

    with mrcfile.open(volume) as mrc:
        volume_data = mrc.data
    if z_range is not None:
        z_range = resolve_z(z_range)
    else:
        z_range = np.unique(coords[:, 2]).astype(int)
    if pre is not None:
        prefix = pre.replace(" ", "") + Path(volume).name.replace('.mrc', '')
    else:
        prefix = Path(volume).name.replace('.mrc', '')

    for z in z_range:
        if z > len(volume_data):
            continue
        # convert to images
        out_name = str( label_dir/f'{prefix}_z{z:02d}.mrc' )
        img_data = volume_data[z - 1]
        if pad > 0:
            img_data = pad_image(img_data, pad)
        with mrcfile.new(out_name, overwrite=True) as mrc:
            mrc.set_data(img_data.astype(np.float32))
        # convert xyz to xy
        out_name = str( label_dir/f'{prefix}_z{z:02d}.thi' )
        write_thi(out_name, coords, z)


def train_main(label_dir, train_out, gpuid='0', sparse=False, epicker_path='epicker_train.sh', \
               model=None, exemplar_dir=None, exemplar_out=None, exemplar_num=0, batchsize=4, epoch=120, lr=1e-4):
    Path(train_out).mkdir(exist_ok=True)
    cmd = f'{epicker_path} --exp_id {train_out} --data {label_dir} --label {label_dir} --train_pct 100 --gpus {gpuid}' 
    if sparse:
        cmd += ' --sparse_anno'
    if model is not None:
        cmd += f' --load_model {model}'
    continual = False
    if exemplar_dir is not None:
        if model is not None:
            continual = True
            cmd += f' --load_exemplar {exemplar_dir}'
        else:
            print("you should provide both model and exemplar to do continual training.")
    if exemplar_out is not None and exemplar_num > 0:
        continual = True
        cmd += f' --output_exemplar {exemplar_out} --sampling_size {exemplar_num}'
    if continual:
        cmd += f' --continual'
    cmd += f' --lr {lr} --batch_size {batchsize} --num_epoch {epoch}'
    print(cmd)
    s = os.system(cmd)
    if s != 0:
        print("exit code", s)
        raise Exception("epicker_train failed, see terminal for detail")


def main(args):
    if args.mode == 'pick':
        if args.model is None:
            raise Exception("model is required")
        if args.volume is None:
            raise Exception("volume is required")
        if args.pick_out is None:
            raise Exception("pick_out is required")
        model:str = absolute_path(args.model)
        volume:str = absolute_path(args.volume)
        z_range:str = args.z_range
        pick_out:str = absolute_path(args.pick_out)
        thres:float = args.thres
        max_num:int = args.max_num
        dist:float = args.dist
        sigma:float = args.sigma
        edgex:int = args.edgex
        edgey:int = args.edgey
        pad:int = args.pad
        gpuid:str = args.gpuid
        epicker_path:str = args.epicker_path
        tmp_in:str = absolute_path(args.tmp_in)
        tmp_out:str = absolute_path(args.tmp_out)
        if epicker_path is None:
            epicker_path = 'epicker.sh'
        pick_main(model, volume, z_range, pick_out, thres, max_num, sigma, dist, edgex, edgey, pad, gpuid, epicker_path, tmp_in, tmp_out)

    elif args.mode == 'label':
        if args.volume is None:
            raise Exception("volume is required")
        if args.label_coord is None:
            raise Exception("label_coord is required")
        if args.label_dir is None:
            raise Exception("label_dir is required")
        volume:str = absolute_path(args.volume)
        label_coord:str = absolute_path(args.label_coord)
        label_dir:str = absolute_path(args.label_dir)
        z_range:str = args.z_range
        pad:int = args.pad
        filt_col:int = args.filt_col
        filt_range:str = args.filt_range
        label_pre:str = args.label_pre
        label_main(volume, z_range, label_coord, label_dir, pad, filt_col, filt_range, label_pre)

    elif args.mode == 'train':
        if args.label_dir is None:
            raise Exception("label_dir is required")
        if args.train_out is None:
            raise Exception("train_out is required")
        label_dir:str = absolute_path(args.label_dir)
        train_out:str = absolute_path(args.train_out)
        gpuid:str = args.gpuid
        sparse:bool = args.sparse
        epicker_path:str = args.epicker_path
        if epicker_path is None:
            epicker_path = 'epicker_train.sh'
        model:str = args.model
        if model is not None:
            model = absolute_path(args.model)
        exemplar_dir:str = args.exemplar_dir
        if exemplar_dir is not None:
            exemplar_dir = absolute_path(args.exemplar_dir)
        exemplar_num:int = args.exemplar_num
        if exemplar_num > 0:
            exemplar_out:str = absolute_path(Path(train_out)/'exemplar')
        else:
            exemplar_out = None
        batchsize:int = args.batchsize
        epoch:int = args.epoch
        lr:float = args.lr
        train_main(label_dir, train_out, gpuid, sparse, epicker_path, model, exemplar_dir, exemplar_out, exemplar_num, batchsize, epoch, lr)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, choices=['pick', 'label', 'train'],
                        help='task you want to do')
    parser.add_argument('--model', type=str,
                        help='the path of model used when pick, or finetune from')
    parser.add_argument('--volume', type=str,
                        help='the mrc file used when pick or generate label')
    parser.add_argument('--z_range', type=str,
                        help='z range for volume when pick or generte label. shuch as "9,12-14"')
    
    parser.add_argument('--pick_out', type=str, default='epicker_pick.txt',
                        help='the file to save pick result, xyz start from 1 and score')
    parser.add_argument('--tmp_in', type=str, default=None,
                        help='if given, using given tmp file and skip epicker')
    parser.add_argument('--tmp_out', type=str, default=None,
                        help='if given, applying epicker on all slices and save all possible coords as tmp file')
    parser.add_argument('--thres', type=float, default = 0.1,
                        help='minimum score to save particle, 0 to 1, default 0.1')
    parser.add_argument('--max_num', type=int, default = 500,
                        help='maximum number to save particle, default 500')
    parser.add_argument('--dist', type=float, default = -1,
                        help='minimum distance(3d) between particles when pick, in pixel')
    parser.add_argument('--sigma', type=float, default = 2,
                        help='apply 3d gauss filter to epicker result, after thres, before dist. default 0 (close), in pixel')
    parser.add_argument('--edgex', type=int, default = 0,
                        help='ignore the particle near edge(x) when pick, in pixel')
    parser.add_argument('--edgey', type=int, default = 0,
                        help='ignore the particle near edge(y) when pick, in pixel')
    parser.add_argument('--pad', type=int, default = -1,
                        help='pad to square with this size when pick and generate label, useful for old epicker, in pixel')
    
    parser.add_argument('--label_coord', type=str,
                        help='coord file used to generate label for train, xyz start from 1')
    parser.add_argument('--filt_col', type=int,
                        help='use this column to filt coord, if coord hs more than 3 column')
    parser.add_argument('--filt_range', type=str,
                        help='require the number(int) of filt column in this range')
    parser.add_argument('--label_pre', type=str,
                        help='add this prefix to the generated label mrc and thi file')
    parser.add_argument('--label_dir', type=str,
                        help='the folder to save label data used when train')
    parser.add_argument('--train_out', type=str,
                        help='the folder to save train result')
    parser.add_argument('--exemplar_dir', type=str,
                        help='you can provide exemplar to do continual training, rather than just finetune')
    parser.add_argument('--exemplar_num', type=int, default=0,
                        help='you can generate exemplar when training, this is sample number in output examplar')
    parser.add_argument('--sparse', action='store_true',
                        help='use sparse label when training')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='batchsize when train, should <= 4*gpu_num, default 4')
    parser.add_argument('--epoch', type=int, default=120,
                        help='num_epoch when train, default 120')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate when train, default 1e-4')
    
    parser.add_argument('--gpuid', type=str, default='0',
                        help='gpu to use when pick or train, default 0')
    parser.add_argument('--epicker_path', type=str,
                        help='you can specify the path of epicker script explicitly, for pick or train')
    
    args = parser.parse_args()
    main(args)