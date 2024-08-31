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
import os
import subprocess
from shutil import rmtree, copyfile, move


def absolute_path(path):
    # path = Path(path) # str or Path
    return os.path.abspath(str(path))


def make_newdir(path, name = "tmp"):
    path = Path(path) # parent path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        print("makedirs:", str(path))
    tmp = path/name
    i = 0
    while True:
        if not tmp.exists():
            break
        i += 1
        name2 = name + "_" + str(i)
        print(name, "exists, try", name2)
        tmp = path/name2
    tmp.mkdir()
    return absolute_path(tmp)


def remove_dir(path):
    path = Path(path)
    if path.is_dir():
        path = absolute_path(path)
        try:
            rmtree(path)
            print("remove", path)
        except Exception as e:
            print(e)
            print("failed to remove", path)


def run_cmd(cmd, path=None):
    # if path is not None:
    #     cwd = os.getcwd()
    #     os.chdir(absolute_path(path))
    # s = os.system(cmd)
    # if path is not None:
    #     os.chdir(cwd)
    process = subprocess.Popen(cmd, shell=True, cwd=path, text=True,
                                stdout=subprocess.PIPE)
    info = []
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if line.find("Iteration") == 0 and line.find(":") > 1:
            try:
                iter_id = int(line.replace("Iteration", "").replace(":", ""))
                if iter_id <= 2 or iter_id % 50 == 1: # print iter 0 1, then print per 50 iters
                    print("\n".join(info))
                info = []
            except:
                pass
        info.append(line)
    print("\n" + "\n".join(info) + "\n") # final iter
    process.wait()
    return process.returncode


def main(args):
    exe=args.exe
    fin=args.fin
    fout=args.fout
    energy=args.energy
    show=args.show
    cut1=args.cut1

    fin = absolute_path(fin)
    if fout[-4:] != ".obj":
        raise Exception("fout should be .obj file")
    out_path = make_newdir(Path(fout).parent, "OptCuts_output")
    if energy <= 4:
        raise Exception("energy should bigger than 4")
    display = True
    if show == 1:
        mode = "10"
    elif show == 2:
        mode = "0"
    else:
        mode = "100"
        display = False
    if cut1:
        CutOption = 1 # long
    else:
        CutOption = 0 # short
    cmd = f"{exe} {mode} {fin} 0.999 1 0 {energy} 1 {CutOption}"
    print(cmd)
    if show == 2: print("Press / to continue OptCuts")

    s = run_cmd(cmd, out_path)
    if s != 0:
        print("exit code", s)
        remove_dir(out_path)
        raise Exception("OptCuts failed, see terminal for detail")
    
    result = list(Path(out_path).glob("output/*/finalResult_mesh.obj"))
    if len(result) == 1:
        result = absolute_path(result[0])
        copyfile(result, fout)
    else:
        raise Exception("fail to find finalResult_mesh.obj in " + out_path)
    
    if display:
        for file in Path(out_path).glob("output/*/*"):
            move(absolute_path(file), out_path)
        remove_dir(Path(out_path)/"output")
        print("check full results in", out_path)
    else:
        remove_dir(out_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A Wrapper of OptCuts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exe', type=str, default="OptCuts_bin",
                        help='the path of your OptCuts')
    parser.add_argument('--fin', type=str, required=True,
                        help='input obj file')
    parser.add_argument('--fout', type=str, required=True,
                        help='output obj file')
    parser.add_argument('--energy', type=float, default=4.1,
                        help='energy bound, should > 4, for example, 5, 4.1, 4.01. smaller energy cause less distortion but more seam')
    parser.add_argument('--cut1', action='store_true',
                        help='set initialCutOption=1, only useful for topological sphere, if default setting failed')
    parser.add_argument('--show', type=int, default=0,
                        help='set 0 to skip, set 1 or 2 to display and save the process of flattening')
    args = parser.parse_args()

    main(args)