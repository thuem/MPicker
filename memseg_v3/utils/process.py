import mrcfile
import numpy as np
import scipy.ndimage as si
# import cc3d
import argparse
import pdb
from tqdm import tqdm

def seg_post_process(args):
    # threshold = param['threshold']
    # lowpass = param['lowpass']
    # vc_cutoff = param['vc_cutoff']
    threshold = args.threshold
    lowpass = args.lowpass
    vc_cutoff = args.voxel_cutoff
    data_path = args.input
    if args.output is None:
        save_path = data_path.replace('.mrc','_post.mrc')
    else:
        save_path = args.output

    with mrcfile.open(data_path) as f:
        raw_seg = f.data
    # smooth
    print('smooth with GF',lowpass)
    if lowpass == 0:
        post_seg = raw_seg
    else:
        post_seg = si.gaussian_filter(raw_seg,lowpass)

    # threshold
    print('threshold with',threshold)
    post_seg = np.where(post_seg > threshold, 1,0)
    # save_path = data_path.replace('.mrc','_smooth_threshold.mrc')
    # with mrcfile.new(save_path, overwrite=True) as f:
    #     f.set_data(post_seg.astype(np.float32))

    # cc 3D
    # print('calculated CC 3D')
    # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    # labels,N = cc3d.connected_components(post_seg, return_N=True, connectivity=6)
    # stats = cc3d.statistics(labels)
    # print('cutoff cc with voxel counts with',vc_cutoff)
    # vc_index = np.argsort(stats['voxel_counts'])
    # vc = np.sort(stats['voxel_counts'])
    # cut_pos = 0
    # for i,v in enumerate(vc):
    #     if v > vc_cutoff:
    #         cut_pos = i
    #         break
    # selected_index =  vc_index[cut_pos:-1]
    # cc_mask = np.zeros_like(labels).astype(np.int64)
    # print('remain cc', selected_index.shape[0])
    # for index in selected_index:
    #     index_mask = np.where(labels == index, 1, 0)
    #     print('total voxel in CC', np.sum(index_mask))
    #     cc_mask += index_mask

    # post_seg = post_seg * cc_mask

    if vc_cutoff > 1:
        print('calculated CC 2D')
        for z in tqdm(range(post_seg.shape[0])):
            # labels, N = cc3d.connected_components(post_seg[z, :, :], return_N=True, connectivity=4)
            labels, N = si.label(post_seg[z, :, :])
            # stats = cc3d.statistics(labels)
            _, voxel_counts = np.unique(labels, return_counts=True)
            cc_mask = np.take(voxel_counts > vc_cutoff, labels)
            # selected_index = np.argwhere(stats['voxel_counts'] > vc_cutoff).flatten()
            # cc_mask = np.zeros_like(labels, dtype=bool)
            # cc_mask[np.isin(labels, selected_index)] = 1
            post_seg[z, :, :][~cc_mask] = 0
        # cc 3D
        print('calculated CC 3D')
        # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        # labels,N = cc3d.connected_components(post_seg, return_N=True, connectivity=6)
        labels,N = si.label(post_seg)
        _, voxel_counts = np.unique(labels, return_counts=True)
        # stats = cc3d.statistics(labels)
        print('cutoff cc with voxel counts with',vc_cutoff * 100)
        vc_index = np.argsort(voxel_counts)
        vc = np.sort(voxel_counts)
        cut_pos = 0
        for i,v in enumerate(vc):
            if v > vc_cutoff * 100:
                cut_pos = i
                break
        selected_index =  vc_index[cut_pos:-1]
        # cc_mask = np.zeros_like(labels, dtype=bool)
        print('remain cc', selected_index.shape[0])
        cc_mask = np.take(voxel_counts > (vc_cutoff * 100), labels)
        print('total voxel in CC:')
        print([voxel_counts[i] for i in selected_index])
        # for index in selected_index:
        #     cc_mask[labels == index] = 1
        #     print('total voxel in CC', stats['voxel_counts'][index])

        post_seg[~cc_mask] = 0

    # save
    # save_path = data_path.replace('.mrc','_post.mrc')
    with mrcfile.new(save_path, overwrite=True) as f:
        f.set_data(post_seg.astype(np.int8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float, default=0.2,
                        help='threshold for generating binary mask from raw prediction')
    parser.add_argument('-l', '--lowpass', type=float, default=0.2,
                        help='lowpass rate for generating binary mask from raw prediction')
    parser.add_argument('-v', '--voxel_cutoff', type=float, default=50,
                        help='Voxel counts cutoff for exclude small noise segmentation')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input raw prediction')
    parser.add_argument('-o', '--output', type=str,
                        help='output 01 mask')
    args = parser.parse_args()
    # param = {'threshold':args.threshold, 'lowpass':args.lowpass, 'vc_cutoff':args.voxel_cutoff}
    seg_post_process(args)
