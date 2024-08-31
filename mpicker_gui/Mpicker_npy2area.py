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
import mrcfile

def get_area(mgrid_surf, length=2):
    # S = sqrt( p(p-a)(p-b)(p-c) ), p=(a+b+c)/2
    length=max(1, int(length))
    ll=length//2
    lr=length-ll
    mydtype = np.float32
    if mgrid_surf.ndim == 3:  # 2d plane, one slice
        is_2d = True
        mgrid_surf = np.expand_dims(mgrid_surf, axis=1)  # add z
    else:
        is_2d = False
    __,__,yshape,xshape=mgrid_surf.shape
    z0, y0, x0 = mgrid_surf[:, :, ll-ll:yshape-lr-ll, ll-ll:xshape-lr-ll]  # x-ll y-ll
    z1, y1, x1 = mgrid_surf[:, :, ll-ll:yshape-lr-ll, ll+lr:xshape-lr+lr]  # x+lr y-ll
    z2, y2, x2 = mgrid_surf[:, :, ll+lr:yshape-lr+lr, ll-ll:xshape-lr-ll]  # x-ll y+lr
    z3, y3, x3 = mgrid_surf[:, :, ll+lr:yshape-lr+lr, ll+lr:xshape-lr+lr]  # x+lr y+lr

    a = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2).astype(mydtype)
    b = np.sqrt((x2 - x0)**2 + (y2 - y0)**2 + (z2 - z0)**2).astype(mydtype)
    c = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2).astype(mydtype)
    p = (a + b + c) / 2
    S = np.sqrt(p * (p - a) * (p - b) * (p - c))
    a = np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2).astype(mydtype)
    b = np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2).astype(mydtype)
    p = (a + b + c) / 2
    S += np.sqrt(p * (p - a) * (p - b) * (p - c))
    areas = np.zeros_like(mgrid_surf[0], dtype=mydtype)
    areas[:, ll:yshape-lr, ll:xshape-lr] = S / length**2

    # not precise but better than 0
    areas[:, yshape-lr:, :] = np.repeat( np.expand_dims(areas[:, yshape-lr-1, :], axis=1), lr, axis=1)  
    areas[:, :ll, :]        = np.repeat( np.expand_dims(areas[:, ll, :],          axis=1), ll, axis=1)
    areas[:, :, xshape-lr:] = np.repeat( np.expand_dims(areas[:, :, xshape-lr-1], axis=2), lr, axis=2)
    areas[:, :, :ll]        = np.repeat( np.expand_dims(areas[:, :, ll],          axis=2), ll, axis=2)
    if is_2d:
        areas = areas[0]
    return areas

def get_stretch(mgrid_surf, length=1):
    # long axis of ellip / short axis of ellip
    length=max(1, int(length))
    ll=length//2 # 0
    lr=length-ll # 1
    if mgrid_surf.ndim == 3:  # 2d plane, one slice
        is_2d = True
        mgrid_surf = np.expand_dims(mgrid_surf, axis=1)  # add z
    else:
        is_2d = False
    __,__,yshape,xshape=mgrid_surf.shape
    z0, y0, x0 = mgrid_surf[:, :, ll-ll:yshape-lr-ll, ll:xshape-lr]        # x-ll  y
    z1, y1, x1 = mgrid_surf[:, :, ll+lr:yshape-lr+lr, ll:xshape-lr]        # x+lr  y
    z2, y2, x2 = mgrid_surf[:, :, ll:yshape-lr,       ll-ll:xshape-lr-ll]  # x     y-ll
    z3, y3, x3 = mgrid_surf[:, :, ll:yshape-lr,       ll+lr:xshape-lr+lr]  # x     y+lr
    a2 = (x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2
    b2 = (x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2
    ab = (x1 - x0)*(x3 - x2) + (y1 - y0)*(y3 - y2) + (z1 - z0)*(z3 - z2)
    # cos_angle = ab / np.sqrt(a2*b2)
    # 2 axis of ellip:  sqrt( ( a^2+b^2 +/- sqrt((a^2-b^2)^2+4c^2) ) / 2 ) , c=ab
    l1 = np.sqrt( a2 + b2 + np.sqrt( (a2-b2)**2 + 4 * ab**2 ) ) / np.sqrt(2)
    l2 = np.sqrt( a2 + b2 - np.sqrt( (a2-b2)**2 + 4 * ab**2 ) ) / np.sqrt(2)
    l2[l2<=0] = l1[l2<=0]*1e-6 # l2 always > 0 unless a paraller to b
    stretch = np.zeros_like(mgrid_surf[0])
    # l1/l2, l2/l1, sqrt(1-(l2/l1)^2), l1/l2+l2/l1-2, which to use ?
    stretch[:, ll:yshape-lr, ll:xshape-lr] = l1 / l2 

    # not precise but better than 0
    stretch[:, yshape-lr:, :] = np.repeat( np.expand_dims(stretch[:, yshape-lr-1, :], axis=1), lr, axis=1)  
    stretch[:, :ll, :]        = np.repeat( np.expand_dims(stretch[:, ll, :],          axis=1), ll, axis=1)
    stretch[:, :, xshape-lr:] = np.repeat( np.expand_dims(stretch[:, :, xshape-lr-1], axis=2), lr, axis=2)
    stretch[:, :, :ll]        = np.repeat( np.expand_dims(stretch[:, :, ll],          axis=2), ll, axis=2)
    if is_2d:
        stretch = stretch[0]
    return stretch

if __name__ == '__main__':
    import sys 
    if len(sys.argv) < 3:
        print("useage:")
        print("script.py input_npy output_mrc")
        print("or script.py input_npy output_mrc {'area', 'stretch'}, default 'area'")
        quit()
    input_npy=sys.argv[1]
    output=sys.argv[2]
    if len(sys.argv) < 4:
        out_type = 'area'
    else:
        out_type = sys.argv[3]

    if out_type == 'stretch':
        get_mrc = get_stretch
    else:
        get_mrc = get_area

    mgrid_surf=np.load(input_npy)

    area_tomo = get_mrc(mgrid_surf)

    with mrcfile.new(output,overwrite=True) as mrc:
        mrc.set_data(area_tomo.astype(np.float32))