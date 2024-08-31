import os
from tqdm import tqdm
import mrcfile
import numpy as np
import argparse
import pdb


def make_finetune_dataset(args):
    with mrcfile.open(args.img_dir) as tomo:
        img_orig = tomo.data[:, :, :].copy()
    if args.seg_dir is not None:
        with mrcfile.open(args.seg_dir) as tomo:
            seg_orig = tomo.data[:, :, :].copy()
    else:
        seg_orig = np.zeros(img_orig.shape)
    if args.coord_dir is not None:
        coord_orig = np.loadtxt(args.coord_dir)
        # coord_orig = np.load(args.coord_dir)
        # coord_orig = np.stack([(coord_orig[:, 4] + coord_orig[:, 5]) // 2, (coord_orig[:, 2] + coord_orig[:, 3]) // 2, (coord_orig[:, 0] + coord_orig[:,1]) // 2]).T
    else:
        coord_orig = None

    x_crop, y_crop = int(args.crop_size.split(",")[0]), int(args.crop_size.split(",")[1])
    x_stride, y_stride = int(args.stride.split(",")[0]), int(args.stride.split(",")[1])
    z_min, z_max = int(args.z_range.split(",")[0]), int(args.z_range.split(",")[1])
    if z_min == -1:
        z_min = 0
    if z_max == -1:
        z_max = img_orig.shape[0]

    img = img_orig[z_min:z_max, :, :]
    seg = seg_orig[z_min:z_max, :, :]

    coord_x = np.arange(img.shape[2] // x_stride) * x_stride + x_crop // 2
    coord_y = np.arange(img.shape[1] // y_stride) * y_stride + y_crop // 2
    X, Y = np.meshgrid(coord_x, coord_y)

    # data_name_split = args.output_dir.split('/')
    # if data_name_split[-1] == '':
    #     data_name = data_name_split[-2]
    # else:
    #     data_name = data_name_split[-1]
    data_name = os.path.basename(args.output_dir)
    if data_name == '':
        data_name = os.path.basename(os.path.dirname(args.output_dir))
    idx = 0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] + x_crop // 2 > img.shape[2]:
                continue
            if Y[i, j] + y_crop // 2 > img.shape[1]:
                continue 

            output_path = os.path.join(args.output_dir, '{}_{}'.format(data_name, idx))
            while os.path.isdir(output_path):
                idx += 1
                output_path = os.path.join(args.output_dir, '{}_{}'.format(data_name, idx))
            os.makedirs(output_path, exist_ok=False)
            img_path = os.path.join(output_path, '{}_{}_{}.mrc'.format(data_name, idx, args.suffix))
            seg_path = os.path.join(output_path, '{}_{}_instance_mask.mrc'.format(data_name, idx))
            
            with mrcfile.new(img_path, overwrite=True) as tomo:
                tomo.set_data(np.float32(img[:, Y[i, j] - y_crop // 2: Y[i, j] + y_crop // 2, X[i, j] - x_crop // 2: X[i, j] + x_crop // 2]))
            with mrcfile.new(seg_path, overwrite=True) as tomo:
                tomo.set_data(np.float32(seg[:, Y[i, j] - y_crop // 2: Y[i, j] + y_crop // 2, X[i, j] - x_crop // 2: X[i, j] + x_crop // 2]))
            
            if coord_orig is not None:
                x_min = coord_orig[:, 0] - (X[i, j] - x_stride // 2) - args.box_size // 2
                x_max = coord_orig[:, 0] - (X[i, j] - x_stride // 2) + args.box_size // 2
                y_min = coord_orig[:, 1] - (Y[i, j] - y_stride // 2) - args.box_size // 2
                y_max = coord_orig[:, 1] - (Y[i, j] - y_stride // 2) + args.box_size // 2
                z_min = coord_orig[:, 2] - args.box_size // 2
                z_max = coord_orig[:, 2] + args.box_size // 2
                coords = np.stack([z_min, z_max, y_min, y_max, x_min, x_max]).T
                label = (x_min >= 0) * (y_min >= 0) * (z_min >= 0) * (x_max < x_crop) * (y_max < y_crop) * (z_max < img.shape[0])
                coords = coords[label == 1, :]
                coords_class = np.zeros(coords.shape[0])
                np.save(os.path.join(output_path, 'boundingbox_coords.npy'.format(data_name, idx)), coords)
                np.save(os.path.join(output_path, 'instance_class.npy'.format(data_name, idx)), coords_class)
            idx = idx + 1

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_dir", type=str, default="/Share/UserHome/hwang/tomo_data/chloroplast_titan3_33000x/c3_chloroplast_sta/tomogram/isonet/corrected_tomos/grid3_5_1_defocus5_Cor2_bin6_flipyz_flipz_corrected.mrc", help="Path of recon image for generated dataset.")
    parser.add_argument("--seg_dir", type=str, default="/Share/Software/deeptomo_dep/memseg_v3/result/test/grid3_5_1_defocus5_Cor2_bin6_flipyz_flipz_corrected_seg_post.mrc", help="Path of segmentation mask for generated dataset. (None when no manual mask)")
    parser.add_argument("--coord_dir", type=str, default=None, help="Path of segmentation mask for generated dataset. (None when no manual mask)")
    parser.add_argument("--output_dir", type=str, default="/Share/Software/deeptomo_dep/memseg_v3/tmp/finetune/", help="Save path of generated dataset.")
    parser.add_argument("--suffix", type=str, default="recon", help="Data suffix for generated dataset.")
    
    parser.add_argument("--seg_only", type=bool, default=True, help="Only generate segmentation data.")
    parser.add_argument("--box_size", type=int, default=20, help="Box size for target protein")
    parser.add_argument("--crop_size", type=str, default="300,300", help="Size of dim XY of a single generated tomogram, should be larger than 150.(In format 'int, int')")
    parser.add_argument("--stride", type=str, default="300,300", help="Stride for cropping generated tomogram (In format 'int, int')")
    parser.add_argument("--z_range", type=str, default="40,120", help="Z range for cropping generated tomogram.(In format 'int, int', -1 for full range of z axis)")

    args = parser.parse_args()
    
    make_finetune_dataset(args)
