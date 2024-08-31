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
from scipy.ndimage import gaussian_filter, binary_dilation
import mrcfile
import argparse


def sample_ratio(mgrid, tilt1, tilt2, gap):
    """angles in radian"""
    z, _, x = mgrid
    num = (tilt2-tilt1)//gap + 1
    r = np.sqrt(x**2 + z**2)
    weight = num/(r*gap*(num-1)+1) # r=0 -> N, r*gap=1 -> 1
    # weight[weight<1] = 1
    return weight


def getctf(mgrid, pixel, defocus, kv=300, cs=0.01, amp=0.07, phaseshift=0):
    z, y, x = mgrid
    defocus = -defocus
    cs = cs * 1e7
    kv = kv * 1e3
    phaseshift = phaseshift * np.pi/180
    pixel = 1/(pixel*len(x))
    r2 = x**2 + y**2 + z**2
    r2 *= pixel**2
    lam = 12.2643247 / np.sqrt(kv * (1 + kv * 0.978466e-6))
    w1 = np.sqrt(1 - amp**2)
    w2 = amp
    k1 = np.pi * lam
    k2 = np.pi/2 * cs * lam**3
    phase = k1 * defocus * r2 + k2 * r2**2 - phaseshift
    ctf = -w1 * np.sin(phase) + w2 * np.cos(phase)
    return ctf


def getmask(mgrid, tilt1, tilt2, sigma=1, cylinder=False, do_cos=False, gap=0):
    z, y, x = mgrid
    tilt1, tilt2, gap = tilt1*np.pi/180, tilt2*np.pi/180, gap*np.pi/180
    angle = np.arctan2(z, x)
    angle[angle<-np.pi/2] += np.pi
    angle[angle>np.pi/2] -= np.pi
    mask = (angle >= tilt1) & (angle <= tilt2)
    if int(sigma) > 0:
        mask = binary_dilation(mask, iterations=int(sigma))
    if cylinder:
        r2 = x**2 + z**2
    else:
        r2 = x**2 + y**2 + z**2
    r2_max = (len(x)//2 - 1)**2
    mask = mask & (r2 < r2_max)
    mask = mask.astype(float)
    if do_cos:
        m = (np.abs(x)>1) | (np.abs(z)>1) # angle is inaccurate for small r
        mask[m] *= np.cos(angle[m])
    if sigma > 0:
        mask = gaussian_filter(mask, sigma)
    if gap > 0:
        mask = mask * np.sqrt(sample_ratio(mgrid, tilt1, tilt2, gap))
    return mask


def main(args):
    defocus = args.df
    pixel = args.pix
    box = args.box
    fout = args.out
    tilt1 = args.t1
    tilt2 = args.t2
    kv = args.kv
    cs = args.cs
    amp = args.amp
    phaseshift = args.shift
    sigma = args.sigma
    cylinder = args.cylinder
    do_cos = args.cos
    do_abs = args.abs
    do_sq = args.sq
    sample = args.sample

    tilt1, tilt2 = min(tilt1, tilt2), max(tilt1, tilt2)
    id_start = -(box//2)
    id_end = id_start + box
    zyx = np.mgrid[id_start:id_end, id_start:id_end, id_start:id_end].astype(float)
    result = getctf(zyx, pixel, defocus, kv, cs, amp, phaseshift)
    if do_abs:
        result = np.abs(result)
    if do_sq:
        result = result**2

    result = result * getmask(zyx, tilt1, tilt2, sigma, cylinder, do_cos, sample)

    with mrcfile.new(fout, overwrite=True) as mrc:
        mrc.set_data(result.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate a simple 3dctf", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--df', type=float, required=True, 
                        help='defocus, in Angstrom')
    parser.add_argument('--pix', type=float, required=True,
                        help='pixel size, in Angstrom')
    parser.add_argument('--box', type=int, required=True,
                        help='box size of output, in pixel (should be same as dxy in Mpicker_2dprojection)')
    parser.add_argument('--out', type=str, required=True,
                        help='output file name')
    parser.add_argument('--t1', type=float, default=-60,
                        help='tilt angle start, in degree')
    parser.add_argument('--t2', type=float, default=60,
                        help='tilt angle end, in degree')
    parser.add_argument('--kv', type=float, default=300,
                        help='voltage, in KiloVolts')
    parser.add_argument('--cs', type=float, default=0.01,
                        help='spherical aberration, in milimeter')
    parser.add_argument('--amp', type=float, default=0.07,
                        help='amplitude contrast')
    parser.add_argument('--shift', type=float, default=0,
                        help='phase shift, in degree')
    parser.add_argument('--sigma', type=float, default=1,
                        help='sigma to do gaussian filter for mask, in pixel')
    parser.add_argument('--cylinder', action='store_true',
                        help='shape of mask, by default use sphere')
    parser.add_argument('--cos', action='store_true',
                        help='multiply cos(tilt) as weight')
    parser.add_argument('--sample', type=float, default=0,
                        help='interval of tilt angle, if > 0, will use weight of sample ratio')
    parser.add_argument('--abs', action='store_true',
                        help='get absolute value of ctf')
    parser.add_argument('--sq', action='store_true',
                        help='get ctf squared')
    
    args = parser.parse_args()
    main(args)