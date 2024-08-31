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
import numpy as np
import scipy.ndimage as nd

import mrcfile
from scipy.optimize import curve_fit, root_scalar, least_squares
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import Rbf, RBFInterpolator
from scipy.ndimage import map_coordinates
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map, thread_map

from mpicker_plot3d import organize_coords, plot_o3d, show_3d, show_3d_texture
from Mpicker_npy2area import get_area, get_stretch
from Mpicker_convert_mrc import write_surface_npz, read_surface_coord
from ellipcylinder import convert2ellipzyx, convertback_ellipzyx, draw_cylinder

import random
from tqdm import tqdm
import warnings
#from skimage import feature
try:
    import cv2
    has_cv2 = True
except:
    print("fail to import opencv, will not use Canny")
    has_cv2 = False
import time
# import pdb

random.seed(3)


def coordxyz2arrayindex(coordxyz, round = False):
    """xyz start from 1, to zyx start from 0"""
    coordxyz = np.array(coordxyz)
    if coordxyz.ndim == 1:
        if round:
            return np.round(coordxyz[::-1]).astype(int) - 1
        else:
            return coordxyz[::-1] - 1
    if round:
        return np.round(coordxyz[:,::-1]).astype(int) - 1
    else:
        return coordxyz[:,::-1] - 1


def arrayindex2coordxyz(arrayindex, round = False):
    """zyx start from 0, to xyz start from 1"""
    arrayindex = np.array(arrayindex)
    if arrayindex.ndim == 1:
        arrayindex = np.expand_dims(arrayindex, axis=0)
    if round:
        return np.round(arrayindex[:,::-1]).astype(int) + 1
    else:
        return arrayindex[:,::-1] + 1


def d_start_end(i0, ni, d, edge=0):
    '''
    real star and end of the slice
    in case d is out of the range of array
    they should be int
    '''
    i_left = min(i0 - edge, d)
    i_right = min(ni - 1 - edge - i0, d)
    return i0 - i_left, i0 + i_right + 1


def curvature_mgrid(fx, fy, fxx, fxy, fyy, mode='gauss'):
    cur_gauss = (fxx * fyy - fxy**2) / (1 + fx**2 + fy**2)**2
    cur_mean = ((1 + fx**2) * fyy + (1 + fy**2) * fxx - 2 * fx * fy * fxy) / (1 + fx**2 + fy**2)**(3 / 2) / 2
    Pmax = cur_mean + np.sqrt(cur_mean**2 - cur_gauss)
    Pmin = cur_mean - np.sqrt(cur_mean**2 - cur_gauss)
    Psmall = np.min([np.abs(Pmin), np.abs(Pmax)], axis=0)
    Pbig = np.max([np.abs(Pmin), np.abs(Pmax)], axis=0)
    if mode == 'mean': return cur_mean
    elif mode == 'max': return Pmax
    elif mode == 'min': return Pmin
    elif mode == 'big': return Pbig
    elif mode == 'small': return Psmall
    else: return cur_gauss


def initialPos(mask, posIn, dist=5):
    '''
    posIn: (x,y,z) in imod
    posOut: (i,j,k) for array[i,j,k]
    '''
    # i0, j0, k0 = int(posIn[2] - 1), int(posIn[1] - 1), int(posIn[0] - 1)
    i0, j0, k0 = coordxyz2arrayindex(posIn, round=True)
    ni, nj, nk = mask.shape
    if i0 > ni - 2 or j0 > nj - 2 or k0 > nk - 2 or min(i0, j0, k0) < 1:  # shouldn't on edge
        #print("Wrong coords!")
        return None
    if mask[i0, j0, k0] != 0:
        posOut = (i0, j0, k0)
    else:
        if dist <= 0:
            #print("No point near!")
            return None
        i1, i2 = d_start_end(i0, ni, dist, 1)
        j1, j2 = d_start_end(j0, nj, dist, 1)
        k1, k2 = d_start_end(k0, nk, dist, 1)
        mask_near = mask[i1:i2, j1:j2, k1:k2].copy()
        mask_near[mask_near != 0] = 1  # just in case
        if mask_near.sum() == 0:
            #print("No point near!")
            return None
        i, j, k = np.ogrid[i1:i2, j1:j2, k1:k2]
        dist_near = (i - i0)**2 + (j - j0)**2 + (k - k0)**2  # distance square
        dist_near = dist_near * mask_near
        dist_near[dist_near == 0] = 3 * dist**2 + 10
        di, dj, dk = np.unravel_index(np.argmin(dist_near), dist_near.shape)
        posOut = (i1 + di, j1 + dj, k1 + dk)
        # print("use nearst point", posOut[2] + 1, posOut[1] + 1, posOut[0] + 1)
    return posOut


def canny_3D(mrc):
    mrc=mrc.astype(np.uint8)
    mrc*=255
    z_len = mrc.shape[0]
    mrc_edge = np.zeros_like(mrc)
    for z in range(z_len):
        edge = cv2.Canny(mrc[z, :, :], 10, 20, True)
        mrc_edge[z] = edge
    return (mrc_edge/255).astype(np.int8)


def get_boundary(mask, near='6'):
    '''
    get boundary by mask-erosion(mask), 26or18or6 near
    then set edge to 0
    '''
    if len(mask[(mask != 1) & (mask != 0)]) > 0:  # just in case
        thres = mask[mask != 0].mean() + mask[mask != 0].std()
        mask2 = np.zeros_like(mask)
        mask2[mask >= thres] = 1
        mask = mask2
        print("you should provide binary mask. now use %.2f as threshold." % thres)
    mask=mask.astype(np.int8)
    if near == 'Canny':
        return canny_3D(mask)
    if near == '26':
        print("erosion")
        mask = nd.binary_erosion(mask).astype(np.int8)
        # struct_ero = np.ones((3, 3, 3))
    if near == '18':
        # struct_ero = np.ones((3, 3, 3))
        # struct_ero[(0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 2, 2, 0, 0, 2, 2), (0, 2, 0, 2, 0, 2, 0, 2)] = 0
        print("dilation")
        mask = nd.binary_dilation(mask).astype(np.int8)
    if near == '6':
        print("default")
    struct_ero = np.zeros((3, 3, 3))
    struct_ero[(1, 0, 2, 1, 1, 1, 1), (1, 1, 1, 0, 2, 1, 1), (1, 1, 1, 1, 1, 0, 2)] = 1
    boundary = mask - nd.binary_erosion(mask, structure=struct_ero).astype(np.int8)
    boundary[[0,-1],:]=0
    boundary[:,[0,-1]]=0
    # boundary = np.zeros(mask.shape)
    # boundary[1:-1, 1:-1, 1:-1] = tmp[1:-1, 1:-1, 1:-1]
    return boundary


def filt_points_cc(points, min_len=5, min_total=10, iscurves=False, curve_xyz='z'):
    p_origin = points
    if iscurves:
        points = list(set([coord for cur in points for coord in cur]))
    points = np.array(points).astype(int)
    if points.ndim != 2:
        print("find surface may be problematic, check the mask or selected point")
        return p_origin
    nz, ny, nx = points[:, 0].max() + 1, points[:, 1].max() + 1, points[:, 2].max() + 1
    tomo = np.zeros((nz, ny, nx))
    for z, y, x in points:
        tomo[z, y, x] = 1
    result = np.zeros_like(tomo)
    result2 = np.zeros_like(tomo)

    for i in range(ny):  # filtzx
        plane = tomo[:, i, :]
        if plane.sum() < min_total:
            continue
        cc2d, N = nd.label(plane, structure=np.ones((3, 3)))
        cc2d_slice = nd.find_objects(cc2d)
        for n in range(N):
            z_range = cc2d_slice[n][0].stop - cc2d_slice[n][0].start
            if z_range >= min_len:
                result[:, i, :][cc2d == n + 1] = 1
    for i in range(nz):  # filtyx
        plane = tomo[i, :, :]
        if plane.sum() < min_total:
            continue
        if min_len < 2:  # ignore cc
            result2[i, :, :][plane > 0] = 1
            continue
        cc2d, N = nd.label(plane, structure=np.ones((3, 3)))
        cc2d_slice = nd.find_objects(cc2d)
        for n in range(N):
            y_range = cc2d_slice[n][0].stop - cc2d_slice[n][0].start
            if y_range >= min_len:
                result2[i, :, :][cc2d == n + 1] = 1
    result *= result2
    if iscurves:
        curves = []
        if curve_xyz == 'z':
            for i in range(result.shape[0]):
                plane = result[i, :, :]
                if plane.sum() > 0:
                    curve = np.argwhere(plane)
                    curve = [(i, p[0], p[1]) for p in curve]
                    curves.append(curve)
        if curve_xyz == 'y':
            for i in range(result.shape[1]):
                plane = result[:, i, :]
                if plane.sum() > 0:
                    curve = np.argwhere(plane)
                    curve = [(p[0], i, p[1]) for p in curve]
                    curves.append(curve)
        if curve_xyz == 'x':
            for i in range(result.shape[2]):
                plane = result[:, :, i]
                if plane.sum() > 0:
                    curve = np.argwhere(plane)
                    curve = [(p[0], p[1], i) for p in curve]
                    curves.append(curve)
        return curves
    result = np.argwhere(result)
    return result


def next_point(plane, posIn, priority, last_pos):
    # here priority's origin is upleft
    i0, j0 = posIn
    ni, nj = plane.shape
    d1 = int((priority.shape[0] - 1) / 2)  # by default 1
    d2 = int((priority.shape[1] - 1) / 2)
    il, ir = min(i0, d1), min(ni - 1 - i0, d1)
    jl, jr = min(j0, d2), min(nj - 1 - j0, d2)
    near = np.zeros_like(priority)
    near[d1 - il:d1 + ir + 1, d2 - jl:d2 + jr + 1] = plane[i0 - il:i0 + ir + 1, j0 - jl:j0 + jr + 1]
    near = near * priority
    i_last, j_last = last_pos
    near[d1 - (i0 - i_last), d2 - (j0 - j_last)] = 0  # next point can't be last point
    if near.sum() == 0:
        return None
    dij = np.argmax(near)
    di = int(dij / priority.shape[1])
    dj = int(dij - di * priority.shape[1])
    posOut = (i0 - il + di, j0 - jl + dj)
    return posOut


def cut_curve_end(curve, idx):
    xmin, xmax = min(curve[0][idx], curve[-1][idx]), max(curve[0][idx], curve[-1][idx])
    new_curve = [point for point in curve if xmin < point[idx] < xmax]
    return new_curve


def get_curve(plane, posIn, priority, noflat=False, cut=0, elongation=np.inf):
    # here priority's origin is upleft
    pos0 = posIn
    curve = [pos0]

    # pri_invert=np.rot90(priority,2) #rotate 180
    # always left to right or always left to right?
    if priority[0, :].sum() * priority[-1, :].sum() == 0:  # up down
        pri_invert = priority[::-1, :]
        flat_id = 0
    elif priority[:, 0].sum() * priority[:, -1].sum() == 0:  # left right
        pri_invert = priority[:, ::-1]
        flat_id = 1

    # z up
    point_sum=0
    pos1 = next_point(plane, pos0, priority, pos0)
    while pos1 != None:
        if noflat and pos1[flat_id] == pos0[flat_id]:
            pass  # will continue but will not add the point
        else:
            curve.append(pos1)
            point_sum+=1
        if point_sum>=elongation:
            break
        pos = next_point(plane, pos1, priority, pos0)
        pos0, pos1 = pos1, pos

    # z down
    point_sum=0
    pos0 = posIn
    pos1 = next_point(plane, pos0, pri_invert, pos0)
    while pos1 != None:
        if noflat and pos1[flat_id] == pos0[flat_id]:
            pass  # will continue but will not add the point
        else:
            curve.insert(0, pos1)
            point_sum+=1
        if point_sum>=elongation:
            break
        pos = next_point(plane, pos1, pri_invert, pos0)
        pos0, pos1 = pos1, pos

    if cut > 0:  # only deal with flat points on 2 ends
        for i in range(cut):
            if len(curve) > 0:
                curve = cut_curve_end(curve, flat_id)

    return curve


priority_default = np.array([[4, 5, 3], [2, 0, 1], [0, 0, 0]])


def select_surf(mask, posIn, ori1="z", ori2="y", priority1=priority_default, priority2=priority_default, elongation=np.inf):
    '''
    scan z in zx plane first to get a curve, then scan y in yx plane for each point on the curve.
    priority(shape should be odd) decides which point will be next point if more than one points near this point, default is
    4 3 2
    5 0 1
    0 0 0
    means the order is clockwise. (here origin is lowerleft and assume you want to search up)
    '''
    pri1 = priority1[::-1, :].copy()  # origin is upleft in numpy array
    pri2 = priority2[::-1, :].copy()
    z0, y0, x0 = posIn
    # first in zx plane
    plane1 = mask[:, y0, :]
    curve1 = get_curve(plane1, (z0, x0), pri1, elongation=elongation)  # [(z,x),...]
    # then in yx plane
    surface_coord = []
    for z1, x1 in curve1:
        plane2 = mask[z1, :, :]
        curve2 = get_curve(plane2, (y0, x1), pri2, elongation=elongation)
        curve2 = [(z1, y2, x2) for y2, x2 in curve2]
        surface_coord += curve2
    return surface_coord


def select_surf_z(mask, curve_y, priority, compare_points=None, compare_num=0, elongation=np.inf):
    '''
    provide one curve in xy, expand each point in it to a curve in xz plane
    [ [z,y,x], ... ] -> [ [ [z,y,x], ... ], ... ] 
    '''
    curves = []
    for z, y, x in curve_y:
        plane = mask[:, y, :]
        curve_z = get_curve(plane, (z, x), priority, elongation=elongation)
        curve_z = [(z0, y, x0) for z0, x0 in curve_z]
        if len(curve_z) < 1: continue
        if compare_points is not None and compare_num > 0:
            points_set = set(curve_z)
            same_set = compare_points.intersection(points_set)
            # len1=len(points_set)
            len2 = len(same_set)
            if len2 >= compare_num:
                curves.append(curve_z)
        else:
            curves.append(curve_z)
    return curves


def select_surf_y(mask, curve_z, priority, elongation=np.inf):
    '''
    provide one curve in xz, expand each point in it to a curve in xy plane
    [ [z,y,x], ... ] -> [ [ [z,y,x], ... ], ... ]
    '''
    curves = []
    for z, y, x in curve_z:
        plane = mask[z, :, :]
        curve_y = get_curve(plane, (y, x), priority, elongation=elongation)
        curve_y = [(z, y0, x0) for y0, x0 in curve_y]
        if len(curve_y) < 1: continue
        curves.append(curve_y)
    return curves


def select_surf_complex1(mask, posIn, ori1="z", ori2="y", priority1=priority_default, priority2=priority_default,
                         pick_num=20, dist=30, edge_ratio=1 / 5, elongation=np.inf):
    '''
    scan z in zx plane first to get a curve, then scan y in yx plane for each point on the curve.
    priority(shape should be odd) decides which point will be next point if more than one points near this point, default is
    4 3 2
    5 0 1
    0 0 0
    means the order is clockwise. (here origin is lowerleft and assume you want to search up)

    new. find intersection of zx first and yx first. then select points do next expand.
    given maximum point number, minimum distance, and how far from edge
    '''
    pri1 = priority1[::-1, :].copy()  # origin is upleft in numpy array
    pri2 = priority2[::-1, :].copy()
    z0, y0, x0 = posIn
    # first in zx plane
    curve1 = select_surf_z(mask, [(z0, y0, x0)], pri1, elongation=elongation)[0]
    # then in yx plane
    curves1 = select_surf_y(mask, curve1, pri2, elongation=elongation)  # [curve,...], curve from small z to big z, each curve from small y to big y

    # first in yx plane
    curve2 = select_surf_y(mask, [(z0, y0, x0)], pri2, elongation=elongation)[0]
    # then in zx plane
    curves2 = select_surf_z(mask, curve2, pri1, elongation=elongation)  # [curve,...], curve from small z to big z, each curve from small y to big y

    points1 = set([coord for cur in curves1 for coord in cur])
    points2 = set([coord for cur in curves2 for coord in cur])
    points_both = list(points1.intersection(points2))
    random.shuffle(points_both)

    # compute edge of each y and z. select point by edge_ratio will need it.
    points_mrc = np.zeros_like(mask)
    for z, y, x in points_both:
        points_mrc[z, y, x] = 1
    z_min, z_max, y_min, y_max = [], [], [], []
    for i in range(mask.shape[1]):
        plane_zx = points_mrc[:, i, :]
        z_list = np.nonzero(plane_zx)[0]
        if len(z_list) > 0:
            z_min.append(z_list.min())
            z_max.append(z_list.max())
        else:
            z_min.append(-1)
            z_max.append(-1)
    for i in range(mask.shape[0]):
        plane_yx = points_mrc[i, :, :]
        y_list = np.nonzero(plane_yx)[0]
        if len(y_list) > 0:
            y_min.append(y_list.min())
            y_max.append(y_list.max())
        else:
            y_min.append(-1)
            y_max.append(-1)

    # select points
    points_selected = [(z0, y0, x0)]
    for p in points_both:
        if len(points_selected) >= pick_num:
            break
        z, y, x = p
        zmin, zmax, ymin, ymax = z_min[y], z_max[y], y_min[z], y_max[z]
        if zmin + edge_ratio * (zmax - zmin) < z < zmax - edge_ratio * (zmax - zmin) and ymin + edge_ratio * (
                ymax - ymin) < y < ymax - edge_ratio * (ymax - ymin):
            for select_p in points_selected:
                if np.linalg.norm(np.array(p) - np.array(select_p)) < dist:
                    break
            else:
                points_selected.append(p)

    # compute intersection surface for each point
    points_final = []
    for p in tqdm(points_selected):
        # first in zx plane
        curve1 = select_surf_z(mask, [p], pri1, elongation=elongation)[0]
        # then in yx plane
        curves1 = select_surf_y(mask, curve1, pri2, elongation=elongation)

        # first in yx plane
        curve2 = select_surf_y(mask, [p], pri2, elongation=elongation)[0]
        # then in zx plane
        curves2 = select_surf_z(mask, curve2, pri1, elongation=elongation)

        points1 = set([coord for cur in curves1 for coord in cur])
        points2 = set([coord for cur in curves2 for coord in cur])
        points_both = list(points1.intersection(points2))
        points_final += points_both
    points_final = list(set(points_final))

    # just plot the point picked
    # tmp=np.zeros_like(mask)
    # for z,y,x in points_selected:
    #     tmp[z,y,x]=1
    # tmp=nd.binary_dilation(tmp,iterations=2)
    # with mrcfile.new('tmp_pick.mrc',overwrite=True) as mrc:
    #     mrc.set_data(tmp.astype(np.float32))

    return points_final


def select_surf_complex2(mask, posIn, ori1="z", ori2="y", priority1=priority_default, priority2=priority_default,
                         pick_num=20, dist=10,
                         ratio1=1 / 5, ratio2=4 / 5, same_num=10, elongation=np.inf):
    '''
    scan z in zx plane first to get a curve, then scan y in yx plane for each point on the curve.
    priority(shape should be odd) decides which point will be next point if more than one points near this point, default is
    4 3 2
    5 0 1
    0 0 0
    means the order is clockwise. (here origin is lowerleft and assume you want to search up)

    new. find some long y curve, then do z expand for them. need filt.
    given maximum curve number, minimum curve distance(z), range from initial point to edge, minimum intersection between new z curve and original surf.
    '''
    pri1 = priority1[::-1, :].copy()  # origin is upleft in numpy array
    pri2 = priority2[::-1, :].copy()
    z0, y0, x0 = posIn
    # first in zx plane
    curve_z = select_surf_z(mask, [(z0, y0, x0)], pri1, elongation=elongation)[0]
    # then in yx plane
    curves_zy = select_surf_y(mask, curve_z, pri2, elongation=elongation)  # [curve,...], curve from small z to big z, each curve from small y to big y
    curves_zy = filt_points_cc(curves_zy, 3, same_num, True, 'z')  # filt by 2d connected component analysis in zx and yx slices.
    points_zy = set([coord for cur in curves_zy for coord in cur])

    z_list = [cur[0][0] for cur in curves_zy]
    zmin, zmax = min(z_list), max(z_list)
    zmiddle = (zmin + zmax) / 2  # it was z0 before, change it
    z_down1, z_down2 = int(zmiddle - ratio1 * (zmiddle - zmin)), int(zmiddle - ratio2 * (zmiddle - zmin))
    z_up1, z_up2 = int(zmiddle + ratio1 * (zmax - zmiddle)), int(zmiddle + ratio2 * (zmax - zmiddle))
    #print('zmin,zdown,zdown,zup,zup,zmax', zmin, z_down2, z_down1, z_up1, z_up2, zmax)

    length_sort = np.array([min(abs(cur[-1][1] - cur[0][1]) + 1, len(cur)) for cur in curves_zy]).argsort()[::-1]
    select_curves_y = select_surf_y(mask, [(z0, y0, x0)], pri2, elongation=elongation)  # [ curve0 ]
    for idx in length_sort:  # long first
        if len(select_curves_y) >= pick_num:
            break
        z = curves_zy[idx][0][0]
        if z_down2 < z < z_down1 or z_up1 < z < z_up2:
            for cur in select_curves_y:
                zs = cur[0][0]
                if abs(z - zs) < dist: break
            else:
                select_curves_y.append(curves_zy[idx])
                #print(z)

    curves_final = []
    for cur in tqdm(select_curves_y):
        curves_yz = select_surf_z(mask, cur, pri1, points_zy, same_num, elongation=elongation)
        # filt_curves_yz(curves_yz)
        curves_yz = filt_points_cc(curves_yz, 3, same_num, True, 'y')
        curves_final += curves_yz
    curves_final += curves_zy

    points_final = list(set([coord for cur in curves_final for coord in cur]))  # delete repeat
    return points_final


def surf_3d(mask, coords):
    surface_mrc = np.zeros_like(mask)
    for z, y, x in coords:
        surface_mrc[z, y, x] = 1
    return surface_mrc


def surf_eq(X, a1, a2, a3, a4, a5, a6):  # z=ax^2+by^2+cxy+dx+ey+f
    y, x = X
    return a1 * x**2 + a2 * y**2 + a3 * x * y + a4 * x + a5 * y + a6


def poly_2d(n):
    def fxy(X, *arg):
        y, x = X
        z = np.zeros_like(x)
        n_arg = 0  # (n+1)*(n+2)/2 in total
        for nx in range(n + 1):
            for ny in range(n + 1):
                if nx + ny <= n:
                    z += arg[n_arg] * x**nx * y**ny
                    n_arg += 1
        return z

    def fdx(X, *arg):
        y, x = X
        dx = np.zeros_like(x)
        n_arg = 0  # (n+1)*(n+2)/2 in total
        for nx in range(n + 1):
            for ny in range(n + 1):
                if nx + ny <= n and nx > 0:
                    dx += arg[n_arg] * nx * x**(nx - 1) * y**ny
                    n_arg += 1
                if nx + ny <= n and nx == 0: n_arg += 1
        return dx

    def fdy(X, *arg):
        y, x = X
        dy = np.zeros_like(x)
        n_arg = 0  # (n+1)*(n+2)/2 in total
        for nx in range(n + 1):
            for ny in range(n + 1):
                if nx + ny <= n and ny > 0:
                    dy += arg[n_arg] * x**nx * ny * y**(ny - 1)
                    n_arg += 1
                if nx + ny <= n and ny == 0: n_arg += 1
        return dy

    return fxy, fdx, fdy


def give_matrix_cylinder(angles):
    # assume the vector to be rotated is [x,y,z] not [z,y,x]
    alpha, beta, gamma = angles
    try:
        rot_matrix = R.from_euler('ZYZ', [alpha, beta, gamma]).as_matrix()  # as_dcm() in old scipy
    except:
        rot_matrix = R.from_euler('ZYZ', [alpha, beta, gamma]).as_dcm()
    return rot_matrix


def poly_cylinder(X, angles, degree=1, polypar=None):
    '''
    z=f(y), F(x,y,z)=z-f(y)=0
    3 angles, n+1 for poly fit
    '''
    # z,y,x=X
    rot_matrix = give_matrix_cylinder(angles).T
    x, y, z = np.dot(rot_matrix, X[::-1])
    if polypar is None:
        polypar = np.polyfit(y, z, deg=degree)
    return z - np.polyval(polypar, y), polypar

def plane2guess(par_plane, n):
    n_vector = par_plane / np.linalg.norm(par_plane)
    theta = np.arccos(n_vector[2])
    phi = np.arctan2(n_vector[1], n_vector[0])
    initial_guess = np.zeros(n + 4)  # 3 euler angle, n order polynominal
    initial_guess[0:3] = phi, theta, 0 #0, -1 * theta, -1 * phi
    initial_guess[-1] = 1 / np.linalg.norm(par_plane)
    #print("initial_guess", initial_guess)
    return initial_guess

def fit_cylinder(coords, n, par_plane):
    initial_guess = plane2guess(par_plane, n)
    rmsd0 = np.sqrt( sum( poly_cylinder(coords.transpose(), initial_guess[:3], polypar=initial_guess[3:])[0]**2 ) / len(coords) )
    lsq_fun = lambda ang: poly_cylinder(coords.transpose(), ang, n, None)[0]
    try:
        warnings.filterwarnings("ignore")
        res = least_squares(lsq_fun, initial_guess[:3])
        warnings.filterwarnings("default")
        angles =res.x
        err, polypar = poly_cylinder(coords.transpose(), angles, n)
        rmsd = np.sqrt( sum(err**2) / len(coords) )
        if rmsd > rmsd0:
            raise Exception()
        par_cylinder = np.hstack([angles, polypar])
    except:
        rmsd = rmsd0
        par_cylinder = initial_guess
        warnings.warn("cylinder fitting failed. try increase (or decrease) cylinder_order. or just set 0 or 1 to close it.")
    print("cylinder fitting rmsd:", rmsd)
    return par_cylinder
    
    

# def curve_fit_pre(coords, initial_guess, n):
#     # fix 2 angles at first
#     fix1, fix2 = initial_guess[:2]
#     initial_guess_short = initial_guess[2:]
#     def poly_cylinder_fix(X, *arg):
#         return poly_cylinder(X, fix1, fix2, *arg)
#     try:
#         par_cylinder, _ = curve_fit(poly_cylinder_fix, coords.transpose(), np.zeros(coords.shape[0]), p0=initial_guess_short,
#                                      maxfev=50 * (n + 2) * (n + 3))
#         initial_guess_new = np.insert(par_cylinder, 0, [fix1, fix2])
#         initial_guess_new[2] = initial_guess_new[2] % np.pi
#         rmsd0 = np.sqrt( sum( poly_cylinder(coords.transpose(),*initial_guess)**2 ) / len(coords) )
#         rmsd = np.sqrt( sum( poly_cylinder(coords.transpose(),*initial_guess_new)**2 ) / len(coords) )
#         if rmsd > rmsd0:
#             raise Exception()
#         else:
#             initial_guess = initial_guess_new
#     except:
#         print("pre cylinder fitting failed")
#     return initial_guess

# def fit_cylinder(coords, n, par_plane, par_plane2=None):
#     '''
#     coords is numpy array, [[z1,y1,x1],[z2,y2,x2]...]
#     '''
#     fail1, fail2 = True, True

#     initial_guess = plane2guess(par_plane, n)
#     initial_guess1 = curve_fit_pre(coords, initial_guess, n)
#     irmsd1 = np.sqrt( sum( poly_cylinder(coords.transpose(),*initial_guess1)**2 ) / len(coords) )
#     try: # plane fitting
#         par_cylinder1, _ = curve_fit(poly_cylinder, coords.transpose(), np.zeros(coords.shape[0]), p0=initial_guess1,
#                                      maxfev=50 * (n + 4) * (n + 5))
#         rmsd1 = np.sqrt( sum( poly_cylinder(coords.transpose(),*par_cylinder1)**2 ) / len(coords) )
#         if rmsd1 > irmsd1 and (initial_guess1 == initial_guess).all():
#             raise Exception()
#         fail1 = False
#     except:
#         if (initial_guess1 == initial_guess).all():
#             print("cylinder fitting failed")
#         else:
#             rmsd1 = irmsd1
#             par_cylinder1 = initial_guess1
#             fail1 = False

#     if par_plane2 is not None:
#         initial_guess = plane2guess(par_plane2, n)
#         initial_guess2 = curve_fit_pre(coords, initial_guess, n)
#         irmsd2 = np.sqrt( sum( poly_cylinder(coords.transpose(),*initial_guess2)**2 ) / len(coords) )
#         try: # max area
#             par_cylinder2, _ = curve_fit(poly_cylinder, coords.transpose(), np.zeros(coords.shape[0]), p0=initial_guess2,
#                                         maxfev=50 * (n + 4) * (n + 5))
#             rmsd2 = np.sqrt( sum( poly_cylinder(coords.transpose(),*par_cylinder2)**2 ) / len(coords) )
#             if rmsd2 > irmsd2 and (initial_guess2 == initial_guess).all():
#                 raise Exception()
#         except:
#             if (initial_guess2 == initial_guess).all():
#                 print("2nd try of cylinder fitting failed")
#             else:
#                 rmsd2 = irmsd2
#                 par_cylinder2 = initial_guess2
#                 fail2 = False

#     if fail1 and fail2:
#         warnings.warn("cylinder fitting failed. try increase (or decrease) cylinder_order. or just set 0 or 1 to close it.")
#         if par_plane2 is not None:
#             return initial_guess2
#         else:
#             return initial_guess1
#     elif not fail1 and not fail2:
#         if rmsd1 > rmsd2:
#             rmsd, par_cylinder = rmsd2, par_cylinder2
#         else:
#             rmsd, par_cylinder = rmsd1, par_cylinder1
#     elif fail1:
#         rmsd, par_cylinder = rmsd2, par_cylinder2
#     else: # rmsd2 is None
#         rmsd, par_cylinder = rmsd1, par_cylinder1

#     print("cylinder fitting rmsd:", rmsd)
#     # initial_guess1[0:3]=par_cylinder[0:3] ##change
#     # return initial_guess1
#     return par_cylinder


def arc_length(x, arg, xmin, length, invert=False):
    '''
    compute arc length for polinominal in cylinder
    used to solve equal arc length interval mgrid
    invert means x < xmin, so quad(...) is negtive
    '''
    p = np.poly1d(arg)
    dp = np.polyder(p)
    dy = lambda x: np.sqrt(1 + dp(x)**2)
    arc = quad(dy, xmin, x)[0]
    dy_x=dy(x)
    if invert: # x < xmin
        arc *= -1
        dy_x *= -1
    y = arc - length  # we want to solve arc=length
    return y, dy_x


# def cylind_ygrid(xmin, xmax, arg, interval=1, expand_ratio=0, expand_y1=0, expand_y2=0):
#     '''
#     solve equal arc length interval
#     expand_ratio, expand in both side. expand_y is in pixel
#     '''
#     total_length = arc_length(xmax, arg, xmin, 0)[0]
#     result = [xmin]
#     x0 = xmin
#     add_left=total_length * expand_ratio + expand_y1*interval
#     add_right=total_length * expand_ratio + expand_y2*interval
#     for length in np.arange(interval, total_length + add_right, interval):
#         root = root_scalar(arc_length, args=(arg, xmin, length), fprime=True, x0=x0)
#         x0 = root.root
#         result.append(root.root)
#     if add_left >= interval:
#         x0 = xmin
#         for length in np.arange(interval, interval + add_left, interval):
#             root = root_scalar(arc_length, args=(arg, xmin, length, True), fprime=True, x0=x0)
#             x0 = root.root
#             result.insert(0, root.root)
#     if add_left <= -1 * interval: # in case expand_ratio < 0 ...
#         cut_length=int(-1 * add_left / interval)
#         if cut_length < len(result):
#             result = result[cut_length:]
#     return np.array(result)


def cylind_ygrid(xmin, xmax, arg, interval=1, expand_ratio=0, expand_y1=0, expand_y2=0, max_scale=0):
    '''
    solve equal arc length interval
    expand_ratio, expand in both side, is for length. expand_y is in pixel
    can set max stretch ratio. max_scale=3 means x[i+1]-x[i]>interval/3
    '''
    total_length = arc_length(xmax, arg, xmin, 0)[0]
    add_left=total_length * expand_ratio # + expand_y1*interval
    add_right=total_length * expand_ratio # + expand_y2*interval
    xmax = root_scalar(arc_length, args=(arg, xmax, add_right), fprime=True, x0=xmax).root
    xmin = root_scalar(arc_length, args=(arg, xmin, add_left, True), fprime=True, x0=xmin).root

    result = [xmin]
    x0 = xmin
    for length in np.arange(interval, total_length, interval):
        root = root_scalar(arc_length, args=(arg, x0, interval), fprime=True, x0=x0)
        x0 = root.root
        if max_scale > 1:
            if x0 - result[-1] < interval/max_scale:
                x0 = result[-1] + interval/max_scale
        result.append(x0)
        if x0 > xmax:
            break

    if expand_y1 >= 1:
        x0 = result[0]
        for i in range(expand_y1):
            root = root_scalar(arc_length, args=(arg, x0, interval, True), fprime=True, x0=x0)
            x0 = root.root
            if max_scale > 1:
                if result[0] - x0 < interval/max_scale:
                    x0 = result[0] - interval/max_scale
            result.insert(0, x0)
    elif expand_y1 <= -1 and expand_y1+len(result) > 0:
        result = result[-expand_y1:]

    if expand_y2 >= 1:
        x0 = result[-1]
        for i in range(expand_y2):
            root = root_scalar(arc_length, args=(arg, x0, interval), fprime=True, x0=x0)
            x0 = root.root
            if max_scale > 1:
                if x0 - result[-1] < interval/max_scale:
                    x0 = result[-1] + interval/max_scale
            result.append(x0)
    elif expand_y2 <= -1 and expand_y2+len(result) > 0:
        result = result[:expand_y2]
    return np.array(result)


def initial_mgrid_cylinder(xmin, xmax, ymin, ymax, thick, poly_arg, interval=1,
                            expand_ratio=0, expand_y1=0, expand_y2=0):
    zgrid = np.arange(-1 * thick, thick + interval, interval).astype(float)
    ygrid = cylind_ygrid(ymin, ymax, poly_arg, interval, expand_ratio, expand_y1, expand_y2).astype(float)
    xmin, xmax = xmin - expand_ratio * (xmax - xmin), xmax + expand_ratio * (xmax - xmin)
    xgrid = np.arange(xmin, xmax + interval, interval).astype(float)
    zgrid = zgrid.reshape(len(zgrid), 1, 1)
    ygrid = ygrid.reshape(1, len(ygrid), 1)
    xgrid = xgrid.reshape(1, 1, len(xgrid))
    mgrid = np.array([zgrid + 0 * ygrid + 0 * xgrid, 0 * zgrid + ygrid + 0 * xgrid, 0 * zgrid + 0 * ygrid + xgrid])
    return mgrid


def plane_eq(X, a, b, c):  # ax+by+cz=1
    z, y, x = X
    return a * x + b * y + c * z


def surf_mgrid_n(yx_mgrid, par_surf, fxy, fdx, fdy):
    '''
    yx_mgrid[:,i,j] is [y,x] in plane xy
    result[:,i,j] is [z,y,x] for point ij
    '''
    y, x = yx_mgrid
    z = fxy(yx_mgrid, *par_surf)
    dx = fdx(yx_mgrid, *par_surf)
    dy = fdy(yx_mgrid, *par_surf)
    d = np.sqrt(dx**2 + dy**2 + 1)
    v = np.array([1 / d, -1 * dy / d, -1 * dx / d])  # z=f(x,y) -> [1,-df/dy,-df/dx]
    return np.array([z, y, x]), v


def give_matrix(abc):
    n_vector = abc / np.linalg.norm(abc)
    theta = np.arccos(n_vector[2])
    phi = np.arctan2(n_vector[1], n_vector[0])
    try:
        rot_matrix = R.from_euler('ZY', [phi, theta]).as_matrix()  # as_dcm() in old scipy
    except:
        rot_matrix = R.from_euler('ZY', [phi, theta]).as_dcm()
    return rot_matrix


def convert_coord(coord, abc, cylinder=False):
    '''
    coord is [[z1,y1,x1],[z2,y2,x2]...] in original tomo
    convert coords to new plane defined by ax+by+cz=1
    just rotate, no translate
    when cylinder=True, abc is angles in give_matrix_cylinder()
    '''
    if cylinder:
        rot_matrix = give_matrix_cylinder(abc)
    else:
        rot_matrix = give_matrix(abc)
    convert_matrix = rot_matrix.T #np.linalg.inv(rot_matrix)
    coord_xyz = np.flip(coord.transpose(), axis=0)  # [[x1,x2,..],[y1,y2,..],[z1,z2,..]]
    coord_xyz_convert = np.dot(convert_matrix, coord_xyz)
    return np.flip(coord_xyz_convert, axis=0).transpose()  # [[z1,y1,x1],[z2,y2,x2]...]


def convert_back_coord(coord, abc, mgrid=False, cylinder=False):
    '''
    coord is the coords for new plane defined by ax+by+cz=1
    convert it back to original coordinate system
    when cylinder=True, abc is angles in give_matrix_cylinder()

    mgrid means coord is np.mgrid[z1:z2,y1:y2,x1:x2] (4D)
    then coord[:,i,j,k]=[i,j,k]
    '''
    if mgrid:
        __, nz, ny, nx = coord.shape
        coord = coord.reshape((3, nz * ny * nx)).transpose()  # convert to [[z1,y1,x1],[z2,y2,x2]...]
    if cylinder:
        convert_matrix = give_matrix_cylinder(abc)
        # convert_matrix = np.linalg.inv(rot_matrix)
    else:
        convert_matrix = give_matrix(abc)
    coord_xyz = np.flip(coord.transpose(), axis=0).astype(float)
    coord_xyz_convert = np.dot(convert_matrix, coord_xyz)
    if mgrid:
        return np.flip(coord_xyz_convert, axis=0).reshape((3, nz, ny, nx))  # convert back
    return np.flip(coord_xyz_convert, axis=0).transpose()


def sample_coord_rbf(coords, par_plane, sample_rate=0.3, dist=10, add_corner_nz=True, cylinder=False):
    # first random pick at least 2000 points, at most 200000 points.
    # par_plane has different meaning with do_cylinder is True or False
    # select points after rotate to new coordinate system
    coords = convert_coord(coords, par_plane, cylinder=cylinder)
    total_coords = coords.shape[0]
    print('total coords num', total_coords)
    pick_num = int(total_coords * sample_rate)
    pick_num = np.clip(pick_num, 2000, 200000)
    pick_num = min(pick_num, total_coords)
    print('pick num', pick_num)
    pick_coords_index = random.sample(range(total_coords), pick_num)
    if add_corner_nz:
        idx_ymin = np.argmin(coords[:, 1])
        idx_ymax = np.argmax(coords[:, 1])
        idx_xmin = np.argmin(coords[:, 2])
        idx_xmax = np.argmax(coords[:, 2])
        idx_xy1 = np.argmin(coords[:, 1] + coords[:, 2])
        idx_xy2 = np.argmax(coords[:, 1] + coords[:, 2])
        idx_xy3 = np.argmin(coords[:, 1] - coords[:, 2])
        idx_xy4 = np.argmax(coords[:, 1] - coords[:, 2])
        pick_coords_index = [idx_ymin, idx_ymax, idx_xmin, idx_xmax, idx_xy1, idx_xy2, idx_xy3, idx_xy4] + pick_coords_index
    pick_coords = coords[pick_coords_index]
    # pick_coords_sparse=[ pick_coords[0] ]
    # for coord in pick_coords:
    #     for select_coord in pick_coords_sparse:
    #         if np.linalg.norm(select_coord[1:]-coord[1:])<dist:#dist in 2d
    #             break
    #     else: # new coord is far to all coord in selected coord
    #         pick_coords_sparse.append(coord)
    # dst = cdist(pick_coords[:, 1:], pick_coords[:, 1:]).astype(np.float16) + np.eye(len(pick_coords), dtype=np.float16) * dist * 10
    # pick_idx = np.ones(len(dst), dtype=bool)
    # for i in range(len(dst)):
    #     if pick_idx[i]:
    #         pick_idx[dst[i] < dist] = False
    # pick_coords_sparse = pick_coords[pick_idx]
    tree = KDTree(pick_coords[:,1:]) # 2d
    pick_idx = np.ones(len(pick_coords), dtype=bool)
    for i in range(len(pick_coords)):
        if pick_idx[i]:
            pick_idx[tree.query_ball_point(pick_coords[i, 1:], dist * 0.9999)] = False # *0.9999 because we allowed ==dist before
            pick_idx[i] = True
    pick_coords_sparse=pick_coords[pick_idx]

    pick_coords_sparse = np.array(pick_coords_sparse)
    print('final pick', len(pick_coords_sparse))
    pick_coords_sparse = convert_back_coord(pick_coords_sparse, par_plane, cylinder=cylinder)
    return pick_coords_sparse


def sample_coord_simple(coords, dist=10, is3d=True, add_corner_nz=True, filt=False, filt_dist=1.9):
    # just sample, not rotate. can be 2d (uv) or 3d (zyx).
    # you can add_corner for 2d. you can filt for 3d.

    total_coords = coords.shape[0]
    print('total coords num', total_coords)
    pick_coords_index = list(range(total_coords))
    if add_corner_nz and not is3d:
        idx_ymin = np.argmin(coords[:, 0])
        idx_ymax = np.argmax(coords[:, 0])
        idx_xmin = np.argmin(coords[:, 1])
        idx_xmax = np.argmax(coords[:, 1])
        idx_xy1 = np.argmin(coords[:, 0] + coords[:, 1])
        idx_xy2 = np.argmax(coords[:, 0] + coords[:, 1])
        idx_xy3 = np.argmin(coords[:, 0] - coords[:, 1])
        idx_xy4 = np.argmax(coords[:, 0] - coords[:, 1])
        pick_coords_index = [idx_ymin, idx_ymax, idx_xmin, idx_xmax, idx_xy1, idx_xy2, idx_xy3, idx_xy4] + pick_coords_index
    pick_coords = coords[pick_coords_index].astype(float)

    # dst = cdist(pick_coords, pick_coords).astype(np.float16) + np.eye(len(pick_coords), dtype=np.float16) * dist * 10
    tree = KDTree(pick_coords)
    pick_idx = np.ones(len(pick_coords), dtype=bool)
    for i in range(len(pick_coords)):
        if pick_idx[i]:
            pick_idx[tree.query_ball_point(pick_coords[i], dist * 0.9999)] = False # *0.9999 because we allowed ==dist before
            pick_idx[i] = True # dist from itself is always 0
    pick_coords_sparse=pick_coords[pick_idx]
    print('final pick', len(pick_coords_sparse))

    if filt and is3d:
        for i, point in enumerate(pick_coords_sparse):
            near_idx = tree.query_ball_point(point, filt_dist)
            near_coords = pick_coords[near_idx]
            pick_coords_sparse[i] = near_coords.mean(axis=0)

    return pick_coords_sparse, np.arange(total_coords)[pick_coords_index][pick_idx]


def nearest_idx(x,x_ref):
    tree=KDTree(x_ref.reshape((len(x_ref),1))) # 1D tree hhh
    new_x=tree.query(x.reshape((len(x),1)), k=1)[1]
    return new_x


def find_xy_ori(coords, yx_mgrid, expand_ratio=0):
    # input is [[z,y,x],...]
    # find a rotation angle so that length in x is minimum
    # return angle and coord in mgrid idx
    coord_xyz = np.flip(coords.transpose(), axis=0).copy()  # [[x1,x2,..],[y1,y2,..],[z1,z2,..]]
    x = yx_mgrid[1, 0, :]
    y = yx_mgrid[0, :, 0]
    coord_xyz[0] = nearest_idx(coord_xyz[0], x)  # convert to "coord" of mgrid
    coord_xyz[1] = nearest_idx(coord_xyz[1], y)

    angles = np.linspace(-1 / 2 * np.pi, 1 / 2 * np.pi, 100)
    x_len = []

    for ang in angles:
        rot_matrix = np.array([[np.cos(ang), -1 * np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        coord_x_convert = np.dot(rot_matrix, coord_xyz[0:2])[0]
        x_len.append(coord_x_convert.max() - coord_x_convert.min())
    x_len = np.array(x_len)
    ang_idx = x_len.argmin()
    angle = angles[ang_idx]

    rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    coord_x_convert, coord_y_convert = np.dot(rot_matrix, coord_xyz[0:2])
    xmin, xmax, ymin, ymax = coord_x_convert.min(), coord_x_convert.max(), coord_y_convert.min(), coord_y_convert.max()
    # x1,x2,x3,x4=coord_x_convert[[xmin_idx,xmax_idx,ymin_idx,ymax_idx]]
    # y1,y2,y3,y4=coord_y_convert[[xmin_idx,xmax_idx,ymin_idx,ymax_idx]]
    if ymax - ymin > 3 * (xmax - xmin): # long and narrow membrane
        xmin, xmax = xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin)
    xmin, xmax = xmin - expand_ratio * (xmax - xmin), xmax + expand_ratio * (xmax - xmin)
    ymin, ymax = ymin - expand_ratio * (ymax - ymin), ymax + expand_ratio * (ymax - ymin)
    corner_xy = np.array([[xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]])
    imatrix = np.array([[np.cos(angle), np.sin(angle)], [-1 * np.sin(angle), np.cos(angle)]])  # convert back
    corner_xy = np.dot(imatrix, corner_xy).astype(int)  # [[x1...x4],[y1...y4]], index for yx_mgrid
    return angle, corner_xy


def rotate_mgrid(initial_mgrid, yx_mgrid, angle_xy, corner_xy):
    corner_mgrid = np.zeros_like(yx_mgrid[0])
    corner_mgrid[corner_xy[1], corner_xy[0]] = 1  # corner is int
    #dist2_corner = (np.ones(4) * interval * 10)**2
    #ij_corner = [0, 0, 0, 0]
    # tree=KDTree(corner_xy.T)
    # [ [xy],[xy]... ; [xy],[xy]... ; ... ]
    # dd,__=tree.query(np.transpose(yx_mgrid[::-1],(1,2,0)),k=1,distance_upper_bound=1.5*interval)
    # corner_mgrid[dd<=interval]=1
    # for i in range(corner_mgrid.shape[0]):  # y
    #     for j in range(corner_mgrid.shape[1]):  # x
    #         x, y = yx_mgrid[1, i, j], yx_mgrid[0, i, j]
    # for k in range(4):  # corner
    #     xc, yc = corner_xy[0][k], corner_xy[1][k]
    #     if abs(x - xc) <= interval and abs(y - yc) <= interval:
    #         if (x - xc)**2 + (y - yc)**2 < dist2_corner[k]:
    #             dist2_corner[k] = (x - xc)**2 + (y - yc)**2
    #             ij_corner[k] = (i, j)
    # for i, j in ij_corner:
    #     corner_mgrid[i, j] = 1
    yx_mgrid_rot = nd.rotate(yx_mgrid, angle_xy, axes=(1, 2), order=1, mode='constant', cval=np.nan, reshape=True, prefilter=False)
    corner_mgrid_rot = nd.rotate(corner_mgrid, angle_xy, order=1, mode='constant', cval=0, reshape=True, prefilter=False)
    y_corner, x_corner = np.nonzero(corner_mgrid_rot)
    # print(np.nonzero(corner_mgrid_rot))
    # yx_mgrid_rot = yx_mgrid_rot[:, y_corner.min() + 1:y_corner.max(), x_corner.min() + 1:x_corner.max()]
    xmin, xmax = x_corner.min() + 1, x_corner.max() - 1
    ymax1 = np.argwhere(np.isnan(yx_mgrid_rot[0, :, xmin]) == False)[:, 0].max()
    ymin1 = np.argwhere(np.isnan(yx_mgrid_rot[0, :, xmin]) == False)[:, 0].min()
    ymax2 = np.argwhere(np.isnan(yx_mgrid_rot[0, :, xmax]) == False)[:, 0].max()
    ymin2 = np.argwhere(np.isnan(yx_mgrid_rot[0, :, xmax]) == False)[:, 0].min()
    ymax = int(min(ymax1, ymax2, y_corner.max() - 1))
    ymin = int(max(ymin1, ymin2, y_corner.min() + 1))
    yx_mgrid_rot = yx_mgrid_rot[:, ymin:ymax + 1, xmin:xmax + 1]
    # plt.imshow(corner_mgrid)
    # plt.show()
    # plt.imshow(corner_mgrid_rot)
    # plt.show()
    # plt.imshow(yx_mgrid[1])
    # plt.show()
    # plt.imshow(yx_mgrid_rot[1])
    # plt.show()
    yx_mgrid = yx_mgrid_rot
    initial_mgrid = np.zeros((3, initial_mgrid.shape[1], yx_mgrid.shape[1], yx_mgrid.shape[2]))  # in fact it just need a right shape...
    return initial_mgrid, yx_mgrid


def numerical_diff_old(z_mgrid, y_mgrid, x_mgrid, spline, interval):
    d = interval / 10
    y_shape, x_shape = z_mgrid.shape
    y_mgrid_dy = y_mgrid + d
    x_mgrid_dx = x_mgrid + d
    z_mgrid_dy = spline(y_mgrid_dy.reshape(y_shape * x_shape), x_mgrid.reshape(y_shape * x_shape)).reshape(y_shape, x_shape)
    z_mgrid_dx = spline(y_mgrid.reshape(y_shape * x_shape), x_mgrid_dx.reshape(y_shape * x_shape)).reshape(y_shape, x_shape)
    dy = (z_mgrid_dy - z_mgrid) / d
    dx = (z_mgrid_dx - z_mgrid) / d
    return dy, dx


def pad_linear(x):
    y_shape, x_shape = x.shape
    x_pad=np.zeros((y_shape+2, x_shape+2))
    x_pad[1:-1, 1:-1]=x.copy()
    x_pad[0,:]=2*x_pad[1,:]-x_pad[2,:]
    x_pad[-1,:]=2*x_pad[-2,:]-x_pad[-3,:]
    x_pad[:,0]=2*x_pad[:,1]-x_pad[:,2]
    x_pad[:,-1]=2*x_pad[:,-2]-x_pad[:,-3]
    return x_pad


def numerical_diff_fast(y_mgrid, x_mgrid, spline, knn=0):
    x, y = x_mgrid, y_mgrid
    y_shape, x_shape = x_mgrid.shape
    z = spline(y.reshape(y_shape * x_shape), x.reshape(y_shape * x_shape)).reshape(y_shape, x_shape)

    knn = int(knn) # number of points when plane fitting
    if knn > 4:
        try:
            import open3d as o3d
            zyx = np.array([z.reshape(-1), y.reshape(-1), x.reshape(-1)]).T
            norm = np.zeros_like(zyx)
            norm[:,0] = 1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(zyx)
            pcd.normals = o3d.utility.Vector3dVector(norm)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn), fast_normal_computation=False)
            vyx = np.array(pcd.normals).T.reshape((3, y_shape, x_shape))
            return z, vyx
        except:
            print("failed when calculate norm vector by open3d")

    x_pad=pad_linear(x)
    y_pad=pad_linear(y)
    z_pad=pad_linear(z)
    dx1=x_pad[1:-1,2:]-x_pad[1:-1,0:-2]
    dy1=y_pad[1:-1,2:]-y_pad[1:-1,0:-2]
    dz1=z_pad[1:-1,2:]-z_pad[1:-1,0:-2]
    dx2=x_pad[2:,1:-1]-x_pad[0:-2,1:-1]
    dy2=y_pad[2:,1:-1]-y_pad[0:-2,1:-1]
    dz2=z_pad[2:,1:-1]-z_pad[0:-2,1:-1]

    det=dx1*dy2-dx2*dy1
    # print("det min", np.abs(det).min())
    fdx = (dy2*dz1-dy1*dz2) / det
    fdy = (dx1*dz2-dx2*dz1) / det

    vyx = np.array([np.ones_like(fdy), -1 * fdy, -1 * fdx])  #z=f(y,x) -> [1,-df/dy,-df/dx]
    vyx = vyx / np.sqrt(vyx[0]**2 + vyx[1]**2 + vyx[2]**2)

    return z, vyx


def numerical_diff_ellip(y_mgrid, x_mgrid, spline, par_ellipcylinder):
    y_shape, x_shape = x_mgrid.shape
    z_mgrid = spline(y_mgrid.flatten(), x_mgrid.flatten()).reshape(y_shape, x_shape)
    zyx, vzyx = convertback_ellipzyx(np.array([z_mgrid, y_mgrid, x_mgrid]), par_ellipcylinder)
    return zyx, vzyx


def norm_mrc(mrc):
    return (mrc - mrc.mean()) / mrc.std() * 100


def spline_mgrid_one(args):
    x_m, y_m, spline, y_shape, x_shape = args
    return spline(y_m.reshape(y_shape * x_shape), x_m.reshape(y_shape * x_shape)).reshape(y_shape, x_shape)


def numerical_diff(y_mgrid, x_mgrid, spline, interval, order=1, n_cpu=3):
    '''
    spline([y,y,y],[x,x,x])=[z,z,z]
    return fdx,fdy,fdxx,fdxy,fdyy
    '''
    x, y = x_mgrid, y_mgrid
    y_shape, x_shape = x_mgrid.shape
    d = interval / 20
    # t0=time.time()
    if n_cpu <= 1:
        spline_mgrid = lambda x_m, y_m: spline(y_m.reshape(y_shape * x_shape), x_m.reshape(y_shape * x_shape)).reshape(y_shape, x_shape)
        z = spline_mgrid(x, y)
        zdx = spline_mgrid(x + d, y)
        zdy = spline_mgrid(x, y + d)
        z_dx = spline_mgrid(x - d, y)
        z_dy = spline_mgrid(x, y - d)
    # t1=time.time()
    else:
        args_spline = [(x + d, y, spline, y_shape, x_shape), (x, y + d, spline, y_shape, x_shape),
                       (x - d, y, spline, y_shape, x_shape), (x, y - d, spline, y_shape, x_shape),
                       (x, y, spline, y_shape, x_shape)]
        pool = ProcessPoolExecutor(max_workers=n_cpu)
        #zdx, zdy, z_dx, z_dy = process_map(spline_mgrid_one, args_spline, max_workers=n_cpu)
        zdx, zdy, z_dx, z_dy, z = pool.map(spline_mgrid_one, args_spline)
    # t2=time.time()
    # print("1cpu:",t1-t0, "ncpu:",t2-t1)
    fdx = (zdx - z_dx) / (2 * d)
    fdy = (zdy - z_dy) / (2 * d)
    vyx = np.array([np.ones_like(fdy), -1 * fdy, -1 * fdx])  #z=f(y,x) -> [1,-df/dy,-df/dx]
    vyx = vyx / np.sqrt(vyx[0]**2 + vyx[1]**2 + vyx[2]**2)
    if order == 2:
        zdxdy = spline_mgrid(x + d, y + d)
        z_dxdy = spline_mgrid(x - d, y + d)
        zdx_dy = spline_mgrid(x + d, y - d)
        z_dx_dy = spline_mgrid(x - d, y - d)
        fdxx = (zdx + z_dx - 2 * z) / d**2
        fdyy = (zdy + z_dy - 2 * z) / d**2
        fdxy = (zdxdy + z_dx_dy - z_dxdy - zdx_dy) / (4 * d**2)
        return z, fdx, fdy, fdxx, fdxy, fdyy
    else:
        return z, vyx


def prepare_initial_yxmgrid(new_coords, interval, thick, rotate=True, cylinder=False, poly_arg=None, expand_ratio=0):
    __, y_min, x_min = np.min(new_coords, axis=0) - interval
    __, y_max, x_max = np.max(new_coords, axis=0) + interval
    if rotate:
        if cylinder:
            initial_mgrid = initial_mgrid_cylinder(x_min, x_max, y_min, y_max, thick, poly_arg, interval=interval, expand_ratio=0, expand_y1=0, expand_y2=0)
            yx_mgrid = initial_mgrid[1:3, 0, :, :]
        else:
            initial_mgrid = np.mgrid[-1 * thick:thick + interval:interval, y_min:y_max + interval:interval, x_min:x_max + interval:interval].astype(float)
            yx_mgrid = np.mgrid[y_min:y_max + interval:interval, x_min:x_max + interval:interval].astype(float)

        angle_xy, corner_xy = find_xy_ori(new_coords, yx_mgrid, expand_ratio)  # [[x x ...],[y y ...]], int
        angle_xy = -1 * angle_xy * 180 / np.pi  # in degree

        x_left, x_right = 0 - corner_xy[0].min() + 4, corner_xy[0].max() - yx_mgrid.shape[2] + 5  # in pixel
        y_left, y_right = 0 - corner_xy[1].min() + 4, corner_xy[1].max() - yx_mgrid.shape[1] + 5
        if cylinder:
            x_min, x_max = x_min - x_left * interval, x_max + x_right * interval
            initial_mgrid = initial_mgrid_cylinder(x_min, x_max, y_min, y_max, thick, poly_arg, interval=interval, expand_ratio=0, expand_y1=y_left, expand_y2=y_right)
            yx_mgrid = initial_mgrid[1:3, 0, :, :]
        else:
            x_min, x_max = x_min - x_left * interval, x_max + x_right * interval
            y_min, y_max = y_min - y_left * interval, y_max + y_right * interval
            initial_mgrid = np.mgrid[-1 * thick:thick + interval:interval, y_min:y_max + interval:interval, x_min:x_max + interval:interval].astype(float)
            yx_mgrid = initial_mgrid[1:3, 0, :, :]
        corner_xy[0] += x_left
        corner_xy[1] += y_left
        initial_mgrid, yx_mgrid = rotate_mgrid(initial_mgrid, yx_mgrid, angle_xy, corner_xy)
    else:
        if cylinder:
            initial_mgrid = initial_mgrid_cylinder(x_min, x_max, y_min, y_max, thick, poly_arg, interval=interval, expand_ratio=expand_ratio, expand_y1=0, expand_y2=0)
            yx_mgrid = initial_mgrid[1:3, 0, :, :]
        else:
            x_min, x_max = x_min - expand_ratio * (x_max - x_min), x_max + expand_ratio * (x_max - x_min)
            y_min, y_max = y_min - expand_ratio * (y_max - y_min), y_max + expand_ratio * (y_max - y_min)
            initial_mgrid = np.mgrid[-1 * thick:thick + interval:interval, y_min:y_max + interval:interval, x_min:x_max + interval:interval].astype(float)
            yx_mgrid = initial_mgrid[1:3, 0, :, :]
    return initial_mgrid, yx_mgrid


def generate_mgrid_rbf(coords, par_plane, thick, interval=1, function='thin_plate_spline', smooth=0, episilon=5,
                        cylinder=False, par_cylinder=None, rotate=True, plot_process=False, all_coord=[[0,0,0]],
                        expand_ratio=0, mean_filter=11, print_time=False, path_rbfcoord=None, par_ellipcylinder=None):
    '''
    generate coords for new tomo that write in mgrid
    mgrid[:,i,j,k] is corresponding [z,y,x] in original tomo for point [i,j,k] in new tomo 
    thick is z depth above and under the plane/curve, you can sum tomo in z axis to get projection 
    '''
    if cylinder:
        par_plane = par_cylinder[0:3]  # 3 euler angles
        poly_arg = par_cylinder[3:]
    else:
        poly_arg=None
    new_coords = convert_coord(coords, par_plane, cylinder=cylinder)
    all_coord = convert_coord(all_coord, par_plane, cylinder=cylinder)

    z_data, y_data, x_data = new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]
    #spline = Rbf(y_data, x_data, z_data, function=function, smooth=smooth, episilon=episilon)
    spline_new = RBFInterpolator(np.array([y_data,x_data]).T, z_data, kernel='thin_plate_spline', smoothing=smooth)
    spline = lambda y,x:  spline_new(np.array([y,x]).T)
    err=np.abs(spline(y_data,x_data)-z_data)
    print("num:",len(z_data),"smooth:",smooth,"rbf rmsd:",np.sqrt((err**2).mean()))
    if smooth==0 and plot_process:
        coords_show,color,size=organize_coords([new_coords,all_coord],[(0,1,0),(0.5,0.5,0.5)],[2,0])
        plot_o3d(coords_show,color,size)

    if smooth>0:
        n_bad=max(1, int(len(new_coords)/10) ) # delete 10% points
        err_sort=err.argsort() # small to big
        new_coords_good=new_coords[err_sort][:len(new_coords)-n_bad]
        if plot_process:
            new_coords_bad=new_coords[err_sort][len(new_coords)-n_bad:]
            coords_show,color,size=organize_coords([new_coords_good,new_coords_bad,all_coord],[(0,1,0),(0,0,1),(0.5,0.5,0.5)],[2,2,0])
            plot_o3d(coords_show,color,size)

        new_coords=new_coords_good
        z_data, y_data, x_data = new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]
        #spline = Rbf(y_data, x_data, z_data, function=function, smooth=smooth, episilon=episilon)
        spline_new = RBFInterpolator(np.array([y_data,x_data]).T, z_data, kernel='thin_plate_spline', smoothing=smooth)
        spline = lambda y,x:  spline_new(np.array([y,x]).T)
        err=np.abs(spline(y_data,x_data)-z_data)
        print("num:",len(z_data),"smooth:",smooth,"rbf rmsd:",np.sqrt((err**2).mean()))

    if path_rbfcoord is not None:
        rbfcoord_xyz = arrayindex2coordxyz( convert_back_coord(new_coords, par_plane, cylinder=cylinder), round=True )
        np.savetxt(path_rbfcoord, rbfcoord_xyz, fmt='%d')

    t0=time.time()
    initial_mgrid, yx_mgrid = prepare_initial_yxmgrid(new_coords, interval, thick, rotate, cylinder, poly_arg, expand_ratio)
    if print_time:
        print("initial mgrid",time.time()-t0)

    y_mgrid, x_mgrid = yx_mgrid
    # y_shape, x_shape = y_mgrid.shape
    # z_mgrid = spline(y_mgrid.reshape(y_shape * x_shape), x_mgrid.reshape(y_shape * x_shape)).reshape(y_shape, x_shape)

    # if not rotate:
    # plt.plot(new_coords[:,2]-x_min,new_coords[:,1]-y_min,'.',markersize=1)
    # plt.plot(x_data-x_min,y_data-y_min,'rx')
    # plt.imshow(z_mgrid)
    # plt.show()

    # dy,dx=np.gradient(z_mgrid,y_mgrid[:,0],x_mgrid[0,:]) #numerical differational.
    # dx,dy,fdxx,fdxy,fdyy=numerical_diff(z_mgrid,y_mgrid,x_mgrid,spline,interval,order=2)
    t0=time.time()
    if par_ellipcylinder is None:
        try:
            z_mgrid, vyx = numerical_diff_fast(y_mgrid, x_mgrid, spline)
        except Exception as e:
            print(e)
            print("numerical_diff_fast fails, use nomal way") # unlikely to happen
            z_mgrid, vyx = numerical_diff(y_mgrid, x_mgrid, spline, interval, order=1) # use spline_new because easy to multiprocess
        fyx = np.array([z_mgrid, y_mgrid, x_mgrid])
    else:
        fyx, vyx = numerical_diff_ellip(y_mgrid, x_mgrid, spline, par_ellipcylinder)

    if print_time:
        print("rbf time",time.time()-t0)
    # curv=curvature_mgrid(dx,dy,fdxx,fdxy,fdyy)
    # with mrcfile.new('dxxdxydyy.mrc',overwrite=True) as mrc:
    #     data=np.array([norm_mrc(fdxx),norm_mrc(fdxy),norm_mrc(fdyy)])
    #     mrc.set_data(data.astype(np.float32))
    # with mrcfile.new('cur.mrc',overwrite=True) as mrc:
    #     data=curv*100
    #     mrc.set_data(data.astype(np.float32))
    # with mrcfile.new('f.mrc',overwrite=True) as mrc:
    #     data=z_mgrid
    #     mrc.set_data(data.astype(np.float32))
    # with mrcfile.new('dxdy.mrc',overwrite=True) as mrc:
    #     data=np.array([norm_mrc(dx),norm_mrc(dy)])
    #     mrc.set_data(data.astype(np.float32))
    if mean_filter>0:
        mean_filter=int(mean_filter)//2*2+1  # odd number
        vyx=nd.uniform_filter(vyx, size=(1,mean_filter,mean_filter))
        vyx = vyx / np.sqrt(vyx[0]**2 + vyx[1]**2 + vyx[2]**2)

    i = 0
    for d in np.arange(-1 * thick, thick + interval, interval):
        initial_mgrid[:, i, :, :] = fyx + d * vyx  ##
        i += 1
    result_mgrid = convert_back_coord(initial_mgrid, par_plane, mgrid=True, cylinder=cylinder)
    tmp = np.zeros((3, 1, vyx.shape[1], vyx.shape[2]))
    tmp[:, 0, :, :] = vyx
    result_vector = convert_back_coord(tmp, par_plane, mgrid=True, cylinder=cylinder)
    return result_mgrid, result_vector[:, 0, :, :]


def generate_mgrid_plane(coords, par_plane, thick, d_plane, interval=1, cylinder=False, rotate=True, expand_ratio=0):
    new_coords = convert_coord(coords, par_plane, cylinder=cylinder)
    # not usefule. use poly fit and n=1 is better.
    poly_arg = None
    initial_mgrid, yx_mgrid = prepare_initial_yxmgrid(new_coords, interval, thick, rotate, cylinder, poly_arg, expand_ratio)
    z_plane = d_plane
    z=np.arange(z_plane - thick, z_plane + thick + interval)
    for i in range(initial_mgrid.shape[1]):
        initial_mgrid[0,i,:,:]=z[i]
        initial_mgrid[1:3,i,:,:]=yx_mgrid
    result_mgrid = convert_back_coord(initial_mgrid, par_plane, mgrid=True, cylinder=cylinder)
    return result_mgrid


def fit_plane_surf_n(coords, fxy, n):
    '''
    fit coords to plane_eq ax+by+cz=1,
    then rotate coords and fit to z=surf_eq(y,x)
    coords is numpy array, [[z1,y1,x1],[z2,y2,x2]...]
    fxy is n-order 2d poly
    '''
    par_plane, __ = curve_fit(plane_eq, coords.transpose(), np.ones(coords.shape[0]))
    new_coords = convert_coord(coords, par_plane)
    par_num = int((n + 1) * (n + 2) / 2)
    par_surf, __ = curve_fit(fxy, new_coords[:, 1:].transpose(), new_coords[:, 0], p0=np.ones(par_num))
    return par_plane, par_surf


def fit_surf_n(coords, fxy, n, par_plane, cylinder=False):
    '''
    rotate coords and fit to z=surf_eq(y,x)
    coords is numpy array, [[z1,y1,x1],[z2,y2,x2]...]
    fxy is n-order 2d poly
    '''
    new_coords = convert_coord(coords, par_plane, cylinder=cylinder)
    par_num = int((n + 1) * (n + 2) / 2)
    initial_guess = np.zeros(par_num)
    initial_guess[0] = new_coords[:, 0].mean()
    # par_surf, __ = curve_fit(fxy, new_coords[:, 1:].transpose(), new_coords[:, 0], p0=initial_guess)
    def jac(X, *arg):
        y, x = X
        j = np.zeros((len(x), par_num))
        n_arg = 0  # (n+1)*(n+2)/2 in total
        for nx in range(n + 1):
            for ny in range(n + 1):
                if nx + ny <= n:
                    j[:, n_arg] = x**nx * y**ny
                    n_arg += 1
        return j
    par_surf, __ = curve_fit(fxy, new_coords[:,1:].transpose(), new_coords[:,0], p0=initial_guess, jac=jac)
    return par_surf


def generate_mgrid_n(coords, par_plane, par_surf, thick, fxy, fdx, fdy, interval=1,
                     cylinder=False, par_cylinder=None, rotate=True, expand_ratio=0,
                     par_ellipcylinder=None):
    '''
    generate coords for new tomo that write in mgrid
    mgrid[:,i,j,k] is corresponding [z,y,x] in original tomo for point [i,j,k] in new tomo 
    thick is z depth above and under the plane/curve, you can sum tomo in z axis to get projection 
    '''
    if cylinder:
        par_plane = par_cylinder[0:3]  # 3 euler angles
        poly_arg = par_cylinder[3:]
    else:
        poly_arg = None
    new_coords = convert_coord(coords, par_plane, cylinder=cylinder)
    initial_mgrid, yx_mgrid = prepare_initial_yxmgrid(new_coords, interval, thick, rotate, cylinder, poly_arg, expand_ratio)

    if par_ellipcylinder is None:
        fyx, vyx = surf_mgrid_n(yx_mgrid, par_surf, fxy, fdx, fdy)
    else:
        y_mgrid, x_mgrid = yx_mgrid
        z_mgrid = fxy(yx_mgrid, *par_surf)
        fyx, vyx = convertback_ellipzyx(np.array([z_mgrid, y_mgrid, x_mgrid]), par_ellipcylinder)

    # dx, dy, fdxx, fdxy, fdyy = numerical_diff(fyx[0], fyx[1], fyx[2], lambda y, x: fxy([y, x], *par_surf), interval, order=2)
    # curv = curvature_mgrid(dx, dy, fdxx, fdxy, fdyy)
    # with mrcfile.new('dxxdxydyy.mrc', overwrite=True) as mrc:
    #     data = np.array([norm_mrc(fdxx), norm_mrc(fdxy), norm_mrc(fdyy)])
    #     mrc.set_data(data.astype(np.float32))
    # with mrcfile.new('cur.mrc', overwrite=True) as mrc:
    #     data = curv * 100
    #     mrc.set_data(data.astype(np.float32))
    # with mrcfile.new('f.mrc', overwrite=True) as mrc:
    #     data = norm_mrc(fyx[0])
    #     mrc.set_data(data.astype(np.float32))
    # with mrcfile.new('dxdy.mrc', overwrite=True) as mrc:
    #     data = np.array([dx, fdx(yx_mgrid, *par_surf), dy, fdy(yx_mgrid, *par_surf)])
    #     mrc.set_data(data.astype(np.float32))

    i = 0
    for d in np.arange(-1 * thick, thick + interval, interval):
        initial_mgrid[:, i, :, :] = fyx + d * vyx  ##
        i += 1
    result_mgrid = convert_back_coord(initial_mgrid, par_plane, mgrid=True, cylinder=cylinder)
    tmp = np.zeros((3, 1, vyx.shape[1], vyx.shape[2]))
    tmp[:, 0, :, :] = vyx
    result_vector = convert_back_coord(tmp, par_plane, mgrid=True, cylinder=cylinder)
    return result_mgrid, result_vector[:, 0, :, :]


def interp_mgrid2tomo(mgrid, tomo_map, fill_value=None, order=1):
    '''
    value of point [i,j,k] in final tomo is the value of point mgrid[:,i,j,k] in original tomo
    if mgrid[:,i,j,k] out of range of origin, fill by mean
    output is in np.float32
    '''
    if fill_value is None:
        fill_value=np.nan
    # nz, ny, nx = tomo_map.shape
    # map_points = (np.arange(0, nz), np.arange(0, ny), np.arange(0, nx))
    # input_points = mgrid.transpose((1, 2, 3, 0))  # so that input[i,j,k,:]=[z,y,x]
    # interp=interpn(map_points,tomo_map,input_points,bounds_error=False,fill_value=fill_value)
    interp=map_coordinates(tomo_map,mgrid,order=order,prefilter=False,cval=fill_value,output=np.float32)
    if np.isnan(interp).any():
        interp[np.isnan(interp)]=interp[~np.isnan(interp)].mean()
    return interp


def extend_surf(mask, surf, iteration=1):
    '''
    make the surf thicker, but still a subset of mask
    '''
    new_surf = surf.copy()
    for i in range(iteration):
        new_surf = mask * nd.binary_dilation(new_surf)
    return new_surf


def transpose_xyz(data, datatype='tomo', ori='x'):
    # datatype tomo or coord, ori x or y or z.
    if datatype == 'coord':
        # [[z,y,x],[z,y,x],...]
        if ori == 'y':
            data = [[z, x, y] for z, y, x in data]
        elif ori == 'z':
            data = [[x, y, z] for z, y, x in data]
        data = np.array(data)
    elif datatype == 'tomo':
        if ori == 'y':
            data = np.transpose(data, (0, 2, 1))
        elif ori == 'z':
            data = np.transpose(data, (2, 1, 0))
    return data


def transpose_back_xyz(data, datatype='tomo', ori='x'):
    # corresponding to transpose_xyz
    if datatype == 'coord':
        if ori == 'y':
            data = [(z, y, x) for z, x, y in data]
        elif ori == 'z':
            data = [(z, y, x) for x, y, z in data]
        else:
            data = [(z, y, x) for z, y, x in data]
    elif datatype == 'tomo':
        if ori == 'y':
            data = np.transpose(data, (0, 2, 1))
        elif ori == 'z':
            data = np.transpose(data, (2, 1, 0))
    return data


def filt_coord_kdtree(coords_zyx, neighbor=50, n_std=2, threads=-1, plot_process=True):
    if len(coords_zyx) < 1000:
        print("points less than 1000, skip filt.")
        return coords_zyx
    # neighbor = int(len(coords_zyx) * neighbor_ratio)
    tree = KDTree(coords_zyx)
    # t1=time.time()
    dist, __ = tree.query(coords_zyx, neighbor, workers=threads)
    # t2=time.time()
    # print(t2-t1)
    dist_mean = dist[:, 1:].mean(axis=1)
    # print(dist_mean.shape)
    thres = dist_mean.mean() + n_std * dist_mean.std()
    # thres2=dist_mean.mean()+2*n_std*dist_mean.std()
    coords_filt = coords_zyx[dist_mean < thres]

    if plot_process:
        # coords_bad1=coords_zyx[(dist_mean>=thres1) & (dist_mean<thres2) ]
        # coords_bad2=coords_zyx[dist_mean>=thres2]
        coords_bad = coords_zyx[dist_mean >= thres]
        coords_show, color, size = organize_coords([coords_filt, coords_bad], [(0, 0.8, 0), (1, 0, 0)])
        #print(len(coords_filt),len(coords_bad1),len(coords_bad2))
        plot_o3d(coords_show, color, size)

    return coords_filt


def coord_global2local(coords,mgrid,tomoshape=None,max_dist=2):
    """
    conver coords in whole tomo to extracted tomo
    coords is [[z,y,x],[z,y,x]...]
    mgrid is from .npy
    output coord in int
    """
    z,y,x=mgrid.reshape((3,-1))

    if tomoshape is not None:
        # filt coord outside tomo, might faster if too outside?
        nz,ny,nx=tomoshape
        delete_idx= (z<0) | (y<0) | (x<0) | (z>=nz) | (y>=ny) | (x>=nx)
        mask=np.ones_like(z,dtype=bool)
        mask[delete_idx]=0
        save_idx=np.argwhere(mask).flatten() # store idx before filt
        z=z[save_idx]
        y=y[save_idx]
        x=x[save_idx]

    # t0=time.time()
    tree = KDTree(np.array([z,y,x]).T, balanced_tree=False) # balanced_tree=False is faster
    dd, ii = tree.query(coords,k=1,distance_upper_bound=max_dist)
    # t1=time.time()
    # print(t1-t0)
    ii = ii[dd<max_dist] # find nearst local point for each global coord

    if tomoshape is not None:
        ii=save_idx[ii] # real idx before filt
        z,y,x=mgrid.reshape((3,-1))

    save_mask=np.zeros_like(z,dtype=bool)
    save_mask[ii]=1
    mz,my,mx = mgrid[0].shape
    save_mask=save_mask.reshape((mz,my,mx))

    mask_edge=np.ones_like(z,dtype=bool)
    if tomoshape is not None:
        mask_edge[delete_idx]=0 # point outside real tomo
        mask_edge=mask_edge.reshape((mz,my,mx))
        mask_edge=nd.binary_erosion(mask_edge,np.ones((3,3,3))) # edge will change to 0 after erosion
    else:
        mask_edge=mask_edge.reshape((mz,my,mx))
        mask_edge[[0,mz-1],:,:]=0
        mask_edge[:,[0,my-1],:]=0
        mask_edge[:,:,[0,mx-1]]=0

    save_mask*=mask_edge

    coord_zyx=np.argwhere(save_mask)
    # print(time.time()-t1)

    return coord_zyx


def coord_global2local_live1(mgrid):
    z, y, x = mgrid.reshape((3, -1))
    tree = KDTree(np.array([z, y, x]).T, balanced_tree=False)  # balanced_tree=False is faster
    return tree


def coord_global2local_live2(coord, mgrid_shape, mgrid_tree, max_dist=2):
    # coord is [z,y,x], mgrid_shape is mgrid.shape=(3,xx,xx,xx)
    dd, ii = mgrid_tree.query(coord, k=1, distance_upper_bound=max_dist)
    if dd >= max_dist:
        return None
    mz, my, mx = mgrid_shape[1:]
    coord_zyx = np.unravel_index(ii, (mz, my, mx))
    z, y, x = coord_zyx
    if z == 0 or z == mz - 1 or y == 0 or y == my - 1 or x == 0 or x == mx - 1:
        return None
    else:
        return np.array(coord_zyx)


def fibonacci_sample(N, half=True):
    gold=(np.sqrt(5)-1)/2
    if half:
        N*=2
        n=np.arange(int(N/2),N+1)
    else:
        n=np.arange(1,N+1)
    theta=np.arccos((2*n-1)/N - 1)
    phi=2*np.pi*gold*n
    return theta, phi

def project_area_one(coords, theta, phi, dist=3):
    # coords is numpy array, [[z1,y1,x1],[z2,y2,x2]...]
    try:
        rot_matrix = R.from_euler('zy', [-phi,-theta]).as_matrix()  # as_dcm() in old scipy
    except:
        rot_matrix = R.from_euler('zy', [-phi,-theta]).as_dcm()
    coord_xyz = np.flip(coords.transpose(), axis=0)  # [[x1,x2,..],[y1,y2,..],[z1,z2,..]]
    coord_xyz_convert = np.dot(rot_matrix, coord_xyz)
    x, y=(coord_xyz_convert[0:2]/dist).astype(int) # 1 point fills dist**2 pixels
    x = x-x.min()
    y = y-y.min()
    xy_grid=np.zeros((x.max()+1,y.max()+1))
    xy_grid[x, y]=1
    return xy_grid.sum()*dist**2

def project_areas(coords, thetas, phis, dist=3):
    result=[]
    for theta,phi in zip(thetas,phis):
        result.append(project_area_one(coords, theta, phi, dist))
    return np.array(result)

def fit_abc_brute(coords,dist=3,N=200):
    # coords is numpy array, [[z1,y1,x1],[z2,y2,x2]...]
    thetas, phis = fibonacci_sample(N)
    areas=project_areas(coords, thetas, phis, dist)
    idx=np.argmax(areas)
    theta=thetas[idx]
    phi=phis[idx]
    a=np.sin(theta)*np.cos(phi)
    b=np.sin(theta)*np.sin(phi)
    c=np.cos(theta)
    z0,y0,x0=np.mean(coords,axis=0) # let mean point go through the plane
    d=a*x0+b*y0+c*z0 # sqrt(a**2+b**2+c**2)=1
    abc=np.array([a,b,c])
    if d==0: # unlikely to happen
        abc = abc / 0.1
    else:
        abc = abc / ( max(0.1,abs(d)) * np.sign(d) ) # abc=abc/d, so that ax+by+cz=1
    return abc


pri_near = np.array([[6, 7, 5],
                     [4, 0, 3],
                     [0, 0, 0]])

pri_near_2 = np.array([[7, 6, 5],
                       [4, 0, 3],
                       [0, 0, 0]])

pri_near_3 = np.array([[6, 5, 4],
                       [7, 0, 3],
                       [0, 0, 0]])

pri_near_4 = np.array([[7, 5, 4],
                       [6, 0, 3],
                       [0, 0, 0]])

pri_nonear4 = np.array([[2, 4, 6, 8, 9, 7, 5, 3, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])

pri_nonear5 = np.array(
    [[2, 4, 6, 8, 10, 11, 9, 7, 5, 3, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def Get_Boundary(mask_path, near_ero = '6'):#, boundaryout_path
    with mrcfile.mmap(mask_path) as mrc:
        tomo_mask = mrc.data
        voxel_size = mrc.voxel_size
    boundary = get_boundary(tomo_mask,near_ero)
    boundary = boundary.astype(np.int8)[:, ::-1, :]
    # tomo_min = np.min(boundary)
    # tomo_max = np.max(boundary)
    # boundary = ((boundary - tomo_min) / (tomo_max - tomo_min) * 255).astype(np.uint8)
    # if boundaryout_path is not None:
    #     with mrcfile.new(boundaryout_path, overwrite=True) as mrc:
    #         mrc.set_data(boundary.astype(np.int8))
    #         mrc.voxel_size = voxel_size
    return boundary


def find_surface_one(args):

    boundary, initial_point, xyz, surf_method, priority1, priority2, left2right, min_surf, elongation_pixel = args

    if not left2right:
        priority1 = priority1[:, ::-1]
        priority2 = priority2[:, ::-1]

    posi = initialPos(boundary, initial_point)
    if posi is None:
        print("no point near", initial_point)
        return np.array([])
    # select surf coord
    boundary = transpose_xyz(boundary, datatype='tomo', ori=xyz)
    posi = transpose_xyz([posi], datatype='coord', ori=xyz)[0]

    if surf_method == 'complex1':
        coord = select_surf_complex1(boundary, posi, priority1=priority1, priority2=priority2,
                                    dist=min_surf * 3, elongation=elongation_pixel)  # first expand in zx then expand in yx
    elif surf_method == 'complex2':
        coord = select_surf_complex2(boundary, posi, priority1=priority1, priority2=priority2,
                                    dist=min_surf, same_num=min_surf, elongation=elongation_pixel)  # first expand in zx then expand in yx
    else:
        coord = select_surf(boundary, posi, priority1=priority1, priority2=priority2,
                            elongation=elongation_pixel)  # first expand in zx then expand in yx

    # coord1 = filt_points_cc(coord1, min_len=1, min_total=min_surf)  # can delete slice with too less point, add it select_surf itself.
    # filt it as point cloud later is better.

    coord = transpose_back_xyz(coord, datatype='coord', ori=xyz)

    return coord


def Find_Surface(boundary_path,mask_path,initial_point,
                 surfout_path,boundaryout_path,
                 near_ero   = '6',
                 priority1  = pri_near_2,
                 priority2  = pri_near_3,
                 left2right = True,
                 xyz        = "x",
                 surf_method= "simple",
                 min_surf   = 10,
                 n_cpu      = 1,
                 elongation_pixel = np.inf):
    # initial_point xyz surf_method left2right elongation_pixel can be list

    if boundary_path is not None:
        with mrcfile.mmap(boundary_path) as mrc:
            boundary = mrc.data
            voxel_size = mrc.voxel_size
    else:
        with mrcfile.mmap(mask_path) as mrc:
            tomo_mask = mrc.data
            voxel_size = mrc.voxel_size
        boundary = get_boundary(tomo_mask, near_ero)
        if boundaryout_path is not None:
            with mrcfile.new(boundaryout_path, overwrite=True) as mrc:
                mrc.set_data(boundary.astype(np.int8))#mrc.set_data(boundary.astype(np.uint8))
                mrc.voxel_size = voxel_size

    boundary = boundary.astype(bool)
    initial_points = initial_point  #dont want to change name...
    if type(initial_points[0]) is int or type(initial_points[0]) is float:
        initial_points = [initial_points]
    xyzs = xyz  #dont want to change name...
    if type(xyzs) is not list:
        xyzs = [xyzs]
    surf_methods = surf_method  #dont want to change name...
    if type(surf_methods) is not list:
        surf_methods = [surf_methods]
    left2rights = left2right  #dont want to change name...
    if type(left2rights) is not list:
        left2rights = [left2rights]
    elongation_pixels = elongation_pixel #dont want to change name...
    if type(elongation_pixels) is not list:
        elongation_pixels = [elongation_pixels]

    if len(initial_points) == 1:
        coords = find_surface_one((boundary, initial_points[0], xyzs[0], surf_methods[0], priority1, priority2, \
                                    left2rights[0], min_surf, elongation_pixels[0]))  # [(zyx),(zyx)...]

    else:
        if len(xyzs) != len(initial_points):
            if len(xyzs) != 1:
                warnings.warn("number of xyz != number of points")
            else:
                xyzs *= len(initial_points)  # broadcast
        if len(surf_methods) != len(initial_points):
            if len(surf_methods) != 1:
                warnings.warn("number of surf_method != number of points")
            else:
                surf_methods *= len(initial_points)
        if len(left2rights) != len(initial_points):
            if len(left2rights) != 1:
                warnings.warn("number of left2right != number of points")
            else:
                left2rights *= len(initial_points)
        if len(elongation_pixels) != len(initial_points):
            if len(elongation_pixels) != 1:
                warnings.warn("number of elongation_pixel != number of points")
            else:
                elongation_pixels *= len(initial_points)
        args_find_surface = [ (boundary, initial_points[i], xyzs[i], surf_methods[i], priority1, priority2, left2rights[i], \
                               min_surf, elongation_pixels[i]) for i in range(len(initial_points)) ]

        n_cpu=min(len(initial_points),n_cpu)
        if n_cpu==1:
            coords_list = thread_map(find_surface_one, args_find_surface, max_workers=1)
        else:
            coords_list = process_map(find_surface_one, args_find_surface, max_workers=n_cpu)

        coords = set()
        for coord in coords_list:
            coords.update(coord)
        coords = list(coords)

    # coord1 = filt_points_cc(coord1, min_len=1, min_total=min_surf)  # can delete slice with too less point, add it select_surf itself.
    # so min_surf is the min_total when do cc_filt, and is min_dist in complex2
    # min_surf should be much smaller than surf size. for example min_surf=10 is suitable if surf size > 100pix.

    if surfout_path is not None:
        # output the generated surface
        # sur = surf_3d(boundary, coords)
        # with mrcfile.new(surfout_path, overwrite=True) as mrc:
        #     mrc.set_data(sur.astype(np.int8))
        #     mrc.voxel_size = voxel_size
        write_surface_npz(boundary.shape, np.array(coords), surfout_path)

    return


def Sample_Points(surf_path,
                  output_path,
                  sample_ratio   = None,
                  rbf_dist       = None,
                  order          = None):
    # sample_ratio(can > 1) will override rbf_dist, dist = sqrt( 1/ ratio), may not accurate.
    # by default, rbf will use rbf_dist, poly fit with n>8 will use ratio=0.1, poly fit with n<=8 will use all
    # output is [ [z,y,x], [z,y,x], ... ], float

    with mrcfile.open(surf_path) as mrc:
        surf_mask = mrc.data
        surf_coord = np.argwhere(surf_mask)  # coord for nonzero point, [[z,y,x],...]

    if sample_ratio is not None:
        dist_sample = np.sqrt( 1/sample_ratio )
    else:  # use default
        if rbf_dist is None: # poly fit
            if order > 8:
                dist_sample = 3
            else:
                dist_sample = 0
        else: # rbf
            dist_sample = rbf_dist

    if dist_sample > 0:
        surf_coord, __ = sample_coord_simple(surf_coord, dist=dist_sample, is3d=True)

    np.save(output_path, surf_coord.astype(np.float16))

    return



def Extract_Surface(surf_path,density_path,
                    output,convert_file,output_plane="just_rotate.mrc",
                    sample_points_path = None,
                    coordsurf       = None,
                    cylinder_order  = None,
                    rbf_dist        = None,
                    order           = None,
                    do_rotate       = True,
                    thick           = None,
                    fill_value      = None,
                    smooth_factor   = 3,
                    plot_process    = False,
                    expand_ratio    = 0,
                    print_time      = False,
                    path_rbfcoord   = None,
                    must_include_file  = None,
                    plot_fitting    = False):

    # use sample_points_path(npy file) first, if is None, then use coordsurf(txt file). or just use all as before(mrc file).
    # output_plane="just_rotate.mrc" ##change
    smooth_factor=10**smooth_factor-1 # map 0 to 0
    t_0=time.time()
    already_sampled = False
    if sample_points_path is not None:
        already_sampled = True
        surf_coord = np.load(sample_points_path)
    elif coordsurf is not None:
        surf_coord = coordxyz2arrayindex( np.loadtxt(coordsurf) )  # zyx
    elif surf_path is not None:
        # with mrcfile.open(surf_path) as mrc:
        #     surf_mask = mrc.data
        # surf_coord = np.argwhere(surf_mask)  # coord for nonzero point
        surf_coord = read_surface_coord(surf_path)
    # else:
    #     surf_coord = np.array(coord)
    t_1=time.time()
    surf_coord = filt_coord_kdtree(surf_coord, plot_process=plot_process) # filt

    if order is not None:
        fxy, fdx, fdy = poly_2d(order)

    t0=time.time()

    par_ellipcylinder = None
    if cylinder_order == -1: # ellipse cylinder
        surf_coord_old = surf_coord.copy()
        surf_coord, par_ellipcylinder = convert2ellipzyx(surf_coord)
        par_plane = np.array([0, 0, 1])
    elif cylinder_order is not None and cylinder_order >= 1 and len(surf_coord) >= 2000:
        par_plane=fit_abc_brute(surf_coord)
    else: # pure plane fitting
        par_plane, __ = curve_fit(plane_eq, surf_coord.transpose(), np.ones(surf_coord.shape[0]))

    if print_time:
        print("plane time",time.time()-t0)
    print("plane", par_plane)
    z_mean, y_mean, x_mean = surf_coord[:, 0].mean(), surf_coord[:, 1].mean(), surf_coord[:, 2].mean()
    d_plane = (par_plane[0] * x_mean + par_plane[1] * y_mean + par_plane[2] * z_mean) / np.linalg.norm(par_plane)  # z of mean point in new coordinate

    if cylinder_order is None:
        do_cylinder = False
        par_cylinder = None
    elif cylinder_order == -1:
        do_cylinder = False
        par_cylinder = None
    elif cylinder_order < 2: # 1 order is meaningless, treat as plane
        do_cylinder = False
        par_cylinder = None
    else:  # polynominal cylinder surface fitting
        do_cylinder = True
        par_cylinder = fit_cylinder(surf_coord, cylinder_order, par_plane)
        par_plane = par_cylinder[0:3]  # par_plane has different meaning with do_cylinder is True or False
        print("cylinder", par_cylinder)
        abc_new = give_matrix_cylinder(par_cylinder[0:3]).T[2]
        d_plane = (abc_new[0] * x_mean + abc_new[1] * y_mean + abc_new[2] * z_mean) / np.linalg.norm(abc_new)
        print("new_plane", abc_new / (abc_new[0] * x_mean + abc_new[1] * y_mean + abc_new[2] * z_mean))

    if rbf_dist is None:  # polynominal fitting
        if already_sampled:
            par_surf = fit_surf_n(surf_coord, fxy, order, par_plane, cylinder=do_cylinder)
        elif order > 8:  # curve fitting will cause too much time if use all point
            surf_coord_sparse = sample_coord_rbf(surf_coord, par_plane, sample_rate=0.3, dist=3, cylinder=do_cylinder)  # about 1/10
            par_surf = fit_surf_n(surf_coord_sparse, fxy, order, par_plane, cylinder=do_cylinder)
        else:
            par_surf = fit_surf_n(surf_coord, fxy, order, par_plane, cylinder=do_cylinder)
        print("surf", par_surf)

    with mrcfile.mmap(density_path) as mrc:
        tomo_density = mrc.data
        voxel_size = mrc.voxel_size

    # get new tomogram by interpolation

    if rbf_dist is not None:  # use rbf interpolation
        if already_sampled:
            # new fitting plane might be a bit different, so sample again to delete points too near.
            sample_rate = 1
            rbf_dist = rbf_dist * 0.7
        else:
            sample_rate = 0.3
        surf_coord_sparse = sample_coord_rbf(surf_coord, par_plane, sample_rate=sample_rate, dist=rbf_dist,
                                             cylinder=do_cylinder)  # rotate, sample, then rotate back

        mgrid_surf, norm_vector = generate_mgrid_rbf(surf_coord_sparse, par_plane, thick,
                                                     function='thin_plate_spline',
                                                     smooth=smooth_factor, episilon=5, cylinder=do_cylinder,
                                                     par_cylinder=par_cylinder, rotate=do_rotate, all_coord=surf_coord,
                                                     plot_process=plot_process,
                                                     expand_ratio=expand_ratio,
                                                     mean_filter=11,
                                                     print_time=print_time,
                                                     path_rbfcoord=path_rbfcoord,
                                                     par_ellipcylinder=par_ellipcylinder)

    else:  # use polynominal fitting
        mgrid_surf, norm_vector = generate_mgrid_n(surf_coord, par_plane, par_surf, thick, fxy, fdx, fdy,
                                                   cylinder=do_cylinder, par_cylinder=par_cylinder,
                                                   rotate=do_rotate,
                                                   expand_ratio=expand_ratio,
                                                   par_ellipcylinder=par_ellipcylinder)  # extend by surf

    # the ralationship between new tomo coords and origin tomo coords is stored in mgrid
    # [i,j,k] in new tomo correspond to mgrid[:,i,j,k]-->[z,y,x](float) in origin tomo

    if plot_fitting and par_ellipcylinder is not None:
        # slice_zyx = mgrid_surf[:,mgrid_surf.shape[1]//2,:,:].reshape(3,-1).T
        cylinder_zyx = draw_cylinder(par_ellipcylinder, tomo_density.shape)
        coords_show, color, size = organize_coords([surf_coord_old, cylinder_zyx], [(0.5,0.5,0.5),(1,1,0.5)], [0,-1])
        plot_o3d(coords_show, color,size)
    if plot_fitting and par_ellipcylinder is None:
        # plot in rotated coord system
        default_plot = True
        if default_plot:
            slice_zyx = mgrid_surf[:,mgrid_surf.shape[1]//2,:,:].reshape(3,-1) # one slice from flatten tomo
            slice_coords = convert_coord(slice_zyx.T, par_plane, cylinder=do_cylinder) # [[z,y,x],[z,y,x],...]
        else:
            slice_zyx = mgrid_surf[:,20,:,:].reshape(3,-1)
            slice_coords = convert_coord(slice_zyx.T, par_plane, cylinder=do_cylinder)
            final_tomo_surf = interp_mgrid2tomo(mgrid_surf, tomo_density, None)[20]
            slice_colors= final_tomo_surf.reshape(-1)
            slice_colors=(np.array([slice_colors,slice_colors,slice_colors]).T-slice_colors.min())/(slice_colors.max()-slice_colors.min())

        new_coords = convert_coord(surf_coord, par_plane, cylinder=do_cylinder)
        __, y_min, x_min = np.min(new_coords, axis=0) - 1
        __, y_max, x_max = np.max(new_coords, axis=0) + 1
        x_min, x_max = x_min - expand_ratio * (x_max - x_min), x_max + expand_ratio * (x_max - x_min)
        y_min, y_max = y_min - expand_ratio * (y_max - y_min), y_max + expand_ratio * (y_max - y_min)
        if x_max - x_min > y_max - y_min:
            y_min -= (x_max-x_min-y_max+y_min)/2
            y_max += (x_max-x_min-y_max+y_min)/2
        else :
            x_min -= (y_max-y_min-x_max+x_min)/2
            x_max += (y_max-y_min-x_max+x_min)/2
        yx_mgrid = np.mgrid[y_min:y_max + 1, x_min:x_max + 1].astype(float)
        y, x = yx_mgrid.reshape(2,-1)
        if not default_plot:
            yy,xx=np.mgrid[y_min-100:y_max + 1+200, x_min+100:x_max+100 + 1].astype(float).reshape(2,-1) ##change
        if do_cylinder:
            z_cylinder_slice = np.polyval(par_cylinder[3:], slice_coords[:,1])
            shift=(z_cylinder_slice - slice_coords[:,0]).max()
            if default_plot:
                z_cylinder = np.polyval(par_cylinder[3:], y) - shift -10 ##change 30
                z_plane = np.ones_like(x) * z_cylinder.min() - 10 ##change xx
                plane_coords = np.array([z_plane, y, x]).T ##change yy xx
            else:
                z_cylinder = np.polyval(par_cylinder[3:], y) - shift - 30
                z_plane = np.ones_like(xx) * z_cylinder.min() - 30
                plane_coords = np.array([z_plane, yy, xx]).T
            cylinder_coords = np.array([z_cylinder, y, x]).T
            if default_plot:
                coords_show,color,size=organize_coords([slice_coords,plane_coords,cylinder_coords],[(0.5,0.5,0.5),(0.9,0.9,0.45),(0.45,0.9,0.9)],[0,-1,-2])
            else:
                coords_show,color,size=organize_coords([plane_coords,cylinder_coords],[(1,1,0.5),(0.5,1,1)],[-1,-2])
                coords_show=np.vstack([coords_show,slice_coords])
                color=np.vstack([color,slice_colors])
                size=np.hstack([size,np.zeros(len(slice_colors))])
            plot_o3d(coords_show,color,size)
        else:
            z_plane = np.ones_like(x) * slice_coords[:,0].min() - 10
            plane_coords = np.array([z_plane, y, x]).T
            coords_show,color,size=organize_coords([slice_coords,plane_coords],[(0.5,0.5,0.5),(1,1,0.5)],[0,-1])
            plot_o3d(coords_show,color,size)

    t_2=time.time()
    y_top = mgrid_surf[1, mgrid_surf.shape[1] // 2, -1, mgrid_surf.shape[3] // 2]
    y_bot = mgrid_surf[1, mgrid_surf.shape[1] // 2, 0, mgrid_surf.shape[3] // 2]
    if y_top < y_bot:
        mgrid_surf = mgrid_surf[:,:,::-1,::-1] # rotate 180
    # if fill_value is None: fill_value=tomo_density.mean()
    if fill_value==0: fill_value=None # not a good input way
    final_tomo_surf = interp_mgrid2tomo(mgrid_surf, tomo_density, fill_value)
    t_3=time.time()

    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(final_tomo_surf)
        mrc.voxel_size = voxel_size

    if output_plane is not None and par_ellipcylinder is None:
        mgrid_plane = generate_mgrid_plane(surf_coord, par_plane, thick*2, d_plane,
                                           cylinder=do_cylinder,
                                           expand_ratio=expand_ratio)  # extend by plane
        y_top = mgrid_plane[1, mgrid_plane.shape[1] // 2, -1, mgrid_plane.shape[3] // 2]
        y_bot = mgrid_plane[1, mgrid_plane.shape[1] // 2, 0, mgrid_plane.shape[3] // 2]
        if y_top < y_bot:
            mgrid_plane = mgrid_plane[:,:,::-1,::-1] # rotate 180
        final_tomo_plane = interp_mgrid2tomo(mgrid_plane, tomo_density, fill_value)
        with mrcfile.new(output_plane, overwrite=True) as mrc:
            mrc.set_data(final_tomo_plane)
            mrc.voxel_size = voxel_size

    # save mgrid so that we can convert coord after
    # original
    # surf_vector = np.array([mgrid_surf[:, 0, :, :], norm_vector])
    # np.save(convert_file, surf_vector)  # assume interval in generate_mgrid is 1
    # change by LSD
    np.save(convert_file, mgrid_surf.astype(np.float32)) # float16 is not enough when plot 3d mesh

    # try:
    #     print("try open3d")
    #     # multiprocess
    #     proc=Process(target=show_3d,args=(output,convert_file,tomo_density.shape))
    #     proc.start()
    #     proc.join()
    # except:
    #     print("open3d failed")
    # show_3d(output,convert_file,tomo_density.shape)
    t_4=time.time()
    if print_time:
        print("load",t_1-t_0)
        print("mgrid",t_2-t_1)
        print("interp",t_3-t_2)
        print("save",t_4-t_3)
        print("total",t_4-t_0)
    return final_tomo_surf, mgrid_surf


def main(args, priority1=pri_near, priority2=pri_near, output_plane="just_rotate.mrc", fill_value=None, near_ero='6',
         left2right=True, do_rotate=True):

    if args.mode == 'surf_expand': # just in case you want to use this script as cmd line version.
        if args.surfout is None:
            args.surfout = 'tmp_surfout.mrc'
        args.surf = args.surfout

    if args.mode in ['surf', 'surf_expand']:
        Find_Surface(boundary_path=args.boundary,
                    mask_path=args.mask,
                    initial_point=[ [int(xyz) for xyz in coord.split(',')] for coord in args.point.split(':') ],
                    surfout_path=args.surfout,
                    boundaryout_path=None,
                    near_ero=near_ero,
                    priority1=priority1,
                    priority2=priority2,
                    left2right=left2right,
                    xyz=args.xyz,
                    surf_method=args.surf_method,
                    min_surf=10)

    if args.mode in ['expand', 'surf_expand']:
        # fit a smooth surface and then expanded tomo along surface
        Sample_Points(surf_path=args.surf,
                      output_path='tmp_sample_points.npy',
                      sample_ratio=None,
                      rbf_dist=args.rbf_dist,
                      order=args.order)

        final_tomo_surf, mgrid_surf=\
        Extract_Surface(surf_path=args.surf,
                        sample_points_path='tmp_sample_points.npy',
                        density_path=args.density,
                        output_plane=output_plane,
                        output=args.output,
                        convert_file=args.convert_file,
                        coordsurf=args.coordsurf,
                        cylinder_order=args.cylinder_order,
                        rbf_dist=args.rbf_dist,
                        order=args.order,
                        do_rotate=do_rotate,
                        thick=args.thick,
                        fill_value=fill_value,
                        smooth_factor=args.smooth)

        # with mrcfile.new('area.mrc',overwrite=True) as mrc:
        #     data=get_area(mgrid_surf)
        #     mrc.set_data(data.astype(np.float32))

    if args.mode == 'convert_coord':
        # convert coordinates in expanded tomogram to coords in original tomogram (3D)
        mgrid_surf = np.load(args.convert_file)
        coord_in = np.loadtxt(args.coordin, ndmin=2)

        ijk_out = np.zeros_like(coord_in, dtype=float)
        for num, xyz in enumerate(coord_in):
            i, j, k = coordxyz2arrayindex(xyz, round=True)
            ijk_out[num] = mgrid_surf[:, i, j, k]
        coord_out = arrayindex2coordxyz(ijk_out)
        np.savetxt(args.coordout, coord_out, fmt='%.2f')

        if args.vectorout is not None and mgrid_surf.shape[1] > 1 :
            vector=mgrid_surf[:,1,:,:]-mgrid_surf[:,0,:,:]
            for num, xyz in enumerate(coord_in):
                i, j, k = coordxyz2arrayindex(xyz, round=True)
                ijk_out[num] = vector[:, j, k]/np.linalg.norm(vector[:, j, k])
            vector_out = ijk_out[:, ::-1]  # dx dy dz
            np.savetxt(args.vectorout, vector_out, fmt='%.3f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Yan Xiaofeng, 20220228")
    parser.add_argument('--mode', type=str, default='surf_expand',
                        choices=['surf', 'expand', 'surf_expand', 'convert_coord'],
                        help='task you want to do')
    parser.add_argument('--mask', type=str,
                        help='mask mrc file of membrane, with only 0 or 1')
    parser.add_argument('--density', type=str,
                        help='density map')
    parser.add_argument('--point', type=str,
                        help='initial point to get the surface, use "x,y,z" in imod coordinate, start from 1')
    parser.add_argument('--xyz', type=str, default='x', choices=['x', 'y', 'z'],
                        help='the oritation you think the surface face to')
    parser.add_argument('--surf_method', type=str, default='simple', choices=['simple', 'complex1', 'complex2'],
                        help='3 ways to find surface, complex way will spend about 30s more')
    parser.add_argument('--thick', type=int, default=5,
                        help='thick above and under the surface, in pixel')
    parser.add_argument('--output', type=str, default='result.mrc',
                        help='the tomogram expanded along surface, nz=2*thick+1')
    parser.add_argument('--convert_file', type=str, default='convert_coord.npy',
                        help='the file used to convert coords in new tomogram to original tomogram')
    parser.add_argument('--boundary', type=str,
                        help='input boundary file, with only 0 or 1, will not use mask')
    parser.add_argument('--surf', type=str,
                        help='input surface file, with only 0 or 1, only useful when task=expand')
    parser.add_argument('--coordsurf', type=str,
                        help='input txt file, record x y z of surface points, start from 0')
    parser.add_argument('--surfout', type=str,
                        help='output surface file, only useful when task=surf or surf_expand')
    parser.add_argument('--coordin', type=str,
                        help='coordinates file of new tomogram, int, only useful when task=convert_coord, start from 1')
    parser.add_argument('--coordout', type=str, default='coords.txt',
                        help='convert to coordinates in original tomogram, float, only useful when task=convert_coord, start from 1')
    parser.add_argument('--vectorout', type=str,
                        help='norm vector for each point, vx vy vz, float, only useful when task=convert_coord and thick>=1')
    parser.add_argument('--order', type=int, default=2,
                        help='order of polynominal when do the curve fitting')
    parser.add_argument('--rbf_dist', type=float,
                        help='use rbf interpolation with this minimum sample distance in pixel, rather than cuve fitting. should not too small, 20 is a good try')
    parser.add_argument('--smooth', type=float, default=0,
                        help='use rbf interpolation with this smooth factor, any number >=0. 10,100,1000... 500 is a good try')
    parser.add_argument('--cylinder_order', type=int,
                        help='order of polynominal when do the curve fitting (6 is a good start). use it when you think the surface is more like a cylindrical surface than a plane')
    args = parser.parse_args()

    main(args)
