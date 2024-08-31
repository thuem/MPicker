import os
import numpy as np
from glob import glob
import mrcfile
import copy
from tqdm import tqdm
import pdb


def generate_dataset(data_list, output_dir, dataset_name, crop_size=None, crop_patch=None):
    img_list = []
    print("Preparing {} dataset...".format(dataset_name))

    for i in tqdm(range(len(data_list))):
        img_path = data_list[i]

        with mrcfile.open(img_path) as tomo:
            img = tomo.data.copy()
        img = img.astype(np.float32)
        
        if crop_patch is None:
            img = (img - np.mean(img)) / np.std(img)
        else:
            center_patch = img[crop_patch[0]:crop_patch[1], crop_patch[2]:crop_patch[3], crop_patch[4]:crop_patch[5]]
            img = (img - np.mean(center_patch)) / np.std(center_patch)
        if crop_size is not None:
            img = img[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3], crop_size[4]:crop_size[5]]

        gt_name = os.path.dirname(img_path)
        seg_set = glob(gt_name + '/*instance_mask*')
        if len(seg_set) is not 0:
            seg_path = seg_set[0]
        else:
            seg_path = ''
        
        bbox_path = os.path.join(gt_name, 'boundingbox_coords.npy')
        class_path = os.path.join(gt_name, 'instance_class.npy')
        
        if os.path.isfile(seg_path):
            with mrcfile.open(seg_path, permissive=True) as tomo:
                seg = tomo.data.copy()
            seg = seg.astype(np.uint8)
            if crop_size is not None:
                seg = seg[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3], crop_size[4]:crop_size[5]]
        else:
            seg = np.array([])
        
        if os.path.isfile(bbox_path):
            box_coord = np.load(bbox_path)
        else:
            box_coord = np.array([])
       
        if os.path.isfile(class_path):
            box_class = np.load(class_path)
            box_class = box_class * 0
        else:
            box_class = np.array([])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # img_out_path = os.path.join(output_dir, '{}_img.npy'.format(gt_name.split('/')[-1]))
        # seg_out_path = os.path.join(output_dir, '{}_seg.npy'.format(gt_name.split('/')[-1]))
        # bbox_out_path = os.path.join(output_dir, '{}_box.npy'.format(gt_name.split('/')[-1]))
        # bclass_out_path = os.path.join(output_dir, '{}_class.npy'.format(gt_name.split('/')[-1]))
        gt_name_basename = os.path.basename(img_path)
        img_out_path = os.path.join(output_dir, '{}_img.npy'.format(gt_name_basename.split("_MWC")[0]))
        seg_out_path = os.path.join(output_dir, '{}_seg.npy'.format(gt_name_basename.split("_MWC")[0]))
        bbox_out_path = os.path.join(output_dir, '{}_box.npy'.format(gt_name_basename.split("_MWC")[0]))
        bclass_out_path = os.path.join(output_dir, '{}_class.npy'.format(gt_name_basename.split("_MWC")[0]))
 
        # Transfer box coords into the form of (x1, y1, z1, x2, y2, z2)
        if len(box_coord) is not 0:
            box_coord = box_coord[:, [4, 2, 0, 5, 3, 1]]

        np.save(img_out_path, img)
        np.save(seg_out_path, seg)
        np.save(bbox_out_path, box_coord)
        np.save(bclass_out_path, box_class)

        img_list.append(img_out_path)

    return img_list