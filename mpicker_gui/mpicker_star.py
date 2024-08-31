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

from typing import Dict, List, Tuple

def read_star_loop(fname: str, block: str = "", once: bool = True) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a STAR file and return the data of specified block.

    Args:
        fname (str): The filename of the STAR file.
        block (str, optional): The block name in the STAR file. Defaults to "".
        once (bool, optional): If True, only return the data of the 1st appeared block. Defaults to True.

    Returns:
        keys (List[str]): The ordered keys of the loop data (of the last block if once=False).
        result (List[Dict[str, str]]): The data in the specified block.
    """
    flag_hit_block = False
    flag_hit_loop = False
    flag_hit_data = False
    result = []
    keys = []
    data_tmp = dict()
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue

            if line.startswith("data_"):
                if flag_hit_block: # already finished a block
                    if once:
                        return keys, result
                    else:
                        flag_hit_block = False
                        flag_hit_loop = False
                        flag_hit_data = False
                        keys = []
                        data_tmp = dict()
                        if line == "data_" + block:
                            flag_hit_block = True
                            print("Another specified block appears")
                        continue
                else:
                    if line == "data_" + block:
                        flag_hit_block = True
                    continue

            if flag_hit_block and not flag_hit_data:
                if line == "loop_":
                    flag_hit_loop = True
                    continue
                if flag_hit_loop:
                    if line.startswith("_"):
                        key = line.split()[0][1:]
                        keys.append(key)
                        data_tmp[key] = ""
                    else:
                        flag_hit_data = True

            if flag_hit_data:
                values = line.split()
                if len(values) != len(keys):
                    raise Exception(f"The number of columns does not match the number of keys:\n{line}")
                data = data_tmp.copy()
                for key, value in zip(keys, values):
                    data[key] = value
                result.append(data)
                continue
    return keys, result

def write_star_loop(keys: List[str], dataList: List[Dict[str, str]], fname: str, block="", overwrite = False):
    if overwrite:
        mode = "w"
    else:
        mode = "a"
    with open(fname, mode, newline="\n") as f: # force to use Linux style newline
        f.write("\n")
        if not overwrite:
            f.write("\n")
        f.write("data_" + block + "\n\n")
        f.write("loop_\n")
        for i,key in enumerate(keys):
            f.write("_" + key + " #" + str(i+1) + "\n")
        for data in dataList:
            values = [data[key] for key in keys]
            f.write("  ".join(values) + "\n")