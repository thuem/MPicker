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
import numpy as np
from pathlib import Path
try:
    import igl
except:
    print("You need to install libigl by pip if you want to do initial mesh parametrization")
    exit()


def write_obj(fout, v, f, uv=None):
    lines = []
    for v1, v2, v3 in v:
        line = f"v {v1} {v2} {v3} \n"
        lines.append(line)
    if uv is not None and len(uv) == len(v):
        for uv1, uv2 in uv:
            line = f"vt {uv1} {uv2} \n"
            lines.append(line)
        for f1, f2, f3 in f+1:
            line = f"f {f1}/{f1} {f2}/{f2} {f3}/{f3} \n"
            lines.append(line)
    else:
        for f1, f2, f3 in f+1:
            line = f"f {f1} {f2} {f3} \n"
            lines.append(line)
    with open(fout, 'w') as fi:
        fi.writelines(lines)


def check_mesh(v, f):
    # just in case
    import open3d as o3d
    # read mesh directly by new open3d might fail if scale loop length??
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    print("before check", mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    num, vid, vnum = igl.connected_components(igl.adjacency_matrix(np.array(mesh.triangles)))
    if num > 1:
        print(f"has {num} connected components, so leave the largest one.")
        id_save = np.argmax(vnum)
        mesh.remove_vertices_by_mask(vid!=id_save)
    print("after check", mesh)
    v = np.array(mesh.vertices, dtype=v.dtype)
    f = np.array(mesh.triangles, dtype=f.dtype)
    return v, f


def all_boundary(f):
    """output a list of all boundary loop (array). longest first."""
    # bound_edges = igl.boundary_facets(f)
    # if len(bound_edges) == 0:
    #     # closed surface
    #     return []
    # bv1, bv2 = bound_edges[np.argsort(bound_edges[:, 0])].T
    # v_2to1 = np.searchsorted(bv1, bv2)
    # bounds = []
    # unsearched = np.ones_like(bv1, dtype=bool)
    # unsearch_id = unsearched.nonzero()[0]
    # while len(unsearch_id) > 0:
    #     curr_id = unsearch_id[0]
    #     unsearched[curr_id] = False
    #     bound_id = [curr_id] # new loop
    #     next_id = v_2to1[curr_id]
    #     while unsearched[next_id]:
    #         curr_id = next_id
    #         unsearched[curr_id] = False
    #         bound_id.append(curr_id)
    #         next_id = v_2to1[curr_id]
    #     bound = bv1[bound_id]
    #     bounds.append(bound)
    #     unsearch_id = unsearched.nonzero()[0]
    bounds = igl.all_boundary_loop(f)
    bounds = [np.array(bnd) for bnd in bounds]
    bounds.sort(reverse=True, key=lambda x:len(x))
    return bounds


def checkuv_invert(f, uv):
    e1 = uv[f[:,1]]-uv[f[:,0]]
    e2 = uv[f[:,2]]-uv[f[:,0]]
    area = e1[:,0] * e2[:,1] - e1[:,1] * e2[:,0]
    return area.min()


def cut_to_disk(v, f):
    cuts = igl.cut_to_disk(f)
    edge2tri = dict()
    for i, (t1,t2,t3) in enumerate(f):
        edge2tri[(t1,t2)] = (i,0)
        edge2tri[(t2,t3)] = (i,1)
        edge2tri[(t3,t1)] = (i,2)
    cut_mask = np.zeros_like(f)
    for cut in cuts:
        print("cut", len(cut))
        for e1,e2 in zip(cut[:-1], cut[1:]):
            tri = edge2tri.get((e1,e2))
            if tri is not None:
                cut_mask[tri] = 1
            tri = edge2tri.get((e2,e1))
            if tri is not None:
                cut_mask[tri] = 1
    v_cut, f_cut = igl.cut_mesh(v, f, cut_mask)
    return v_cut, f_cut


def bnd_length(v, bnd):
    edges = v[bnd[1:]] - v[bnd[:-1]]
    length = np.linalg.norm(edges, axis=1).sum()
    if bnd[0] != bnd[-1]:
        length += np.linalg.norm(v[bnd[0]]-v[bnd[-1]])
    return length


def mesh_parametrizations(v, fs, bnd):
    bnd_uv = igl.map_vertices_to_circle(v, bnd) # why unit circle??
    radius = np.linalg.norm(bnd_uv[0])
    scale = bnd_length(v, bnd) / (2*np.pi*radius)
    bnd_uv *= scale
    areas = []
    uvs = []
    for f in fs:
        try:
            uv = igl.harmonic_uniform_laplacian(f, bnd, bnd_uv, 1)
        except:
            uv = igl.harmonic_weights_uniform_laplacian(f, bnd, bnd_uv, 1)
        uvs.append(uv)
        areas.append(checkuv_invert(f, uv))
        try:
            uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
        except:
            uv = igl.harmonic_weights(v, f, bnd, bnd_uv, 1)
        uvs.append(uv)
        areas.append(checkuv_invert(f, uv))
    idx = np.array(areas).argmax()
    area = areas[idx]
    print("area", area)
    if area <= 0:
        return None
    else:
        return uvs[idx]


def fill_holes(f, bnds):
    f_add = []
    for bnd in bnds:
        v1 = bnd[0]
        for v2, v3 in zip(bnd[2:], bnd[1:-1]):
            f_add.append([v1,v2,v3])
    f_add = np.array(f_add, dtype=f.dtype)
    f_fake = np.vstack((f, f_add))
    return f_fake


def parametrization_main(v, f, cutdisk=False, fill=False):
    """Only for disk"""
    uv = None
    f_fake = None
    bnds = all_boundary(f)
    if len(bnds) == 0:
        print("Closed surface, skipping. Optcuts can deal with it.")
        return v, f, uv
    elif len(bnds) == 1:
        print("Surface with only one boundary.")
        uv = mesh_parametrizations(v, [f], bnds[0])
        return v, f, uv
    else:
        print("find", len(bnds), "boundaries.")
        if cutdisk:
            v, f = cut_to_disk(v, f)
            bnds = all_boundary(f)
            if len(bnds) == 1:
                print("Cut to topological disk.")
                uv = mesh_parametrizations(v, [f], bnds[0])
                return v, f, uv
            else:
                print("Still not topological disk after cut, result might be wrong")
                print("more than one connected components? you can try --check")
                if len(bnds) == 0:
                    return v, f, uv

        for i in range(len(bnds)):
            bnd = bnds[i]
            if len(bnd) <= 1:
                continue
            print("try boundary", i)
            bnds_fill = bnds[:i] + bnds[i+1:]
            f_fake = fill_holes(f, bnds_fill)
            uv = mesh_parametrizations(v, [f, f_fake], bnd)
            if uv is not None:
                break
        if fill and f_fake is not None:
            print("fill the holes for output obj")
            if uv is None:
                f_fake = fill_holes(f, bnds[1:])
            return v, f_fake, uv
        else:
            return v, f, uv
        

def main(args):
    fin = args.fin
    fout = args.fout
    cutdisk = args.cutdisk
    fill = args.fill
    check = args.check
    no_uv = args.no_uv

    v, f = igl.read_triangle_mesh(fin)
    if check:
        v, f = check_mesh(v, f)
    v, f, uv = parametrization_main(v, f, cutdisk, fill)
    if uv is None:
        print("parametrization fails, output mesh without texture coordinates")
        print("you can try --cutdisk and --check")
    else:
        print("parametrization succeeds")
        if no_uv:
            print("output mesh without texture coordinates")
            uv = None

    if len(fout) < 4 or fout[-4:] != ".obj":
        fout = fout + ".obj"
    Path(fout).parent.mkdir(parents=True, exist_ok=True)
    write_obj(fout, v, f, uv)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get initial mesh parametrization for open surface. Require libigl.", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fin', type=str, required=True,
                        help='input file of triangle mesh')
    parser.add_argument('--fout', type=str, required=True,
                        help='output obj file')
    parser.add_argument('--cutdisk', action='store_true',
                        help='cut the mesh into topological disk at first. You can try it if default set failed.')
    parser.add_argument('--fill', action='store_true',
                        help='fill holes in naive way if the surface has more than one boundaries.')
    parser.add_argument('--check', action='store_true',
                        help='check input mesh at first, such as remove duplicated_vertices, non_manifold_edges.')
    parser.add_argument('--no_uv', action='store_true',
                        help='not write texture coordinates in output file.')
    args = parser.parse_args()

    main(args)
