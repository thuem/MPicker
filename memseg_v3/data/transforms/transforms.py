# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from skimage.transform import resize

from .utils import coord_trans
from .utils import bbox_spatial_trans
from .utils import interpolate_img
from .utils import create_zero_centered_coordinate_mesh
from .utils import elastic_deform_coordinates
from .utils import rotate_coords_3d
from .utils import scale_coords

import pdb



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seg, box_coord, box_class):
        for t in self.transforms:
            img, seg, box_coord, box_class = t(img, seg, box_coord, box_class)
        return img, seg, box_coord, box_class

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
class Consistence(object):
    def __init__(self, transforms, keep):
        self.transforms = transforms
        self.keep = keep

    def __call__(self, img, seg, box_coord, box_class):
        for idx, t in enumerate(self.transforms):
            img, seg, box_coord, box_class = t(img, seg, box_coord, box_class)
            if idx == self.keep:
                self.img_weak = img
        return self.img_weak, img, seg, box_coord, box_class

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class PrecropTransform(object):
    def __init__(self, crop_size, random_crop):
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __call__(self, img, seg, box_coord, box_class):

        img = np.transpose(img, [2, 1, 0])
        if len(seg) is not 0:
            seg = np.transpose(seg, [2, 1, 0])
        crop_dims = range(len(img.shape))

        coord_bias = []

        if self.random_crop:
            crop_center = {ii: np.random.randint(low=self.crop_size[ii]//2,
                            high=img.shape[ii] - self.crop_size[ii]//2) for ii in crop_dims}
        else:
            crop_center = {ii: img.shape[ii] // 2 for ii in crop_dims}

        for ii in crop_dims:
            min_crop = int(crop_center[ii] - self.crop_size[ii] // 2)
            max_crop = int(crop_center[ii] + self.crop_size[ii] // 2)
            img = np.take(img, indices=range(min_crop, max_crop), axis=ii)
            if len(seg) is not 0:
                seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii)
                seg[seg > 1] = 1
                seg[seg < 1] = 0
            
            coord_bias.append(min_crop)
            coord_bias.append(max_crop)

        if len(box_coord) is not 0:
            box_coord, box_class = coord_trans(box_coord, box_class, coord_bias)
        else:
            box_class = np.array([])

        return img, seg, box_coord, box_class


class SpatialTransform(object):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
    """
    def __init__(self, patch_size, random_crop=True,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.), p_el_per_sample=1,
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi), p_rot_per_sample=1,
                 do_scale=True, scale=(0.75, 1.25), p_scale_per_sample=1,
                 border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0):
                 
        self.patch_size = patch_size
        self.random_crop = random_crop
        
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.p_el_per_sample = p_el_per_sample
        
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.p_rot_per_sample = p_rot_per_sample
        
        self.do_scale = do_scale
        self.scale = scale
        self.p_scale_per_sample = p_scale_per_sample

        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
    
    def __call__(self, img, seg, box_coord, box_class):
        # generate coord map
        coords = create_zero_centered_coordinate_mesh(self.patch_size)

        # generate spatial transform parameters
        if np.random.uniform() < self.p_el_per_sample and self.do_elastic_deform:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)

        if np.random.uniform() < self.p_rot_per_sample and self.do_rotation:
            if self.angle_x[0] == self.angle_x[1]:
                a_x = self.angle_x[0]
            else:
                a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            if self.angle_y[0] == self.angle_y[1]:
                a_y = self.angle_y[0]
            else:
                a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            if self.angle_z[0] == self.angle_z[1]:
                a_z = self.angle_z[0]
            else:
                a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            a_x = 0
            a_y = 0
            a_z = 0

        if np.random.uniform() < self.p_scale_per_sample and self.do_scale:
            if np.random.random() < 0.5 and self.scale[0] < 1:
                sc = np.random.uniform(self.scale[0], 1)
            else:
                sc = np.random.uniform(max(self.scale[0], 1), self.scale[1])
            coords = scale_coords(coords, sc)
        else:
            sc = 1

        center = []
        for d in range(len(img.shape)):
            if self.random_crop:
                ctr = int(np.random.uniform(self.patch_size[d] / 2., 
                                            img.shape[d] - self.patch_size[d] / 2.))
            else:
                ctr = int(np.round(img.shape[d] / 2.))
            coords[d] += ctr
            center.append(ctr)

        img_result = interpolate_img(img, coords, self.order_data,
                              self.border_mode_data, cval=self.border_cval_data)

        seg_result = np.zeros(self.patch_size)
        if len(seg) is not 0:
            seg_result = interpolate_img(seg, coords, self.order_seg,
                                         self.border_mode_seg, cval=self.border_cval_seg, is_seg=True)
            # for d in range(0, int(seg.max())):
 
                # v = seg.max() - d
                # seg_trans = np.float32(seg * (seg == v))
                # seg_trans = interpolate_img(seg_trans, coords, self.order_seg, self.border_mode_seg, 
                #                             cval=self.border_cval_seg)
                # seg_trans[seg_trans > 0] = v
                # seg_trans[seg_trans < 0] = 0
                # mask = 1 - (seg_trans > 0)
                # seg_result += mask * np.int16(seg_trans)
        
        if len(box_coord) is not 0:
            box_result, class_result = bbox_spatial_trans(box_coord, box_class, center, self.patch_size, [-a_x, -a_y, -a_z], 1/sc)
        else:
            box_result = np.array([])
            class_result = np.array([])

        return img_result, seg_result, box_result, class_result



class GaussianNoiseTransform(object):
    """Adds additive Gaussian Noise

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """

    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.noise_variance = noise_variance

    def __call__(self, img, seg, box_coord, box_class):
        if np.random.uniform() < self.p_per_sample:
            if self.noise_variance[0] == self.noise_variance[1]:
                variance = self.noise_variance[0]
            else:
                variance = random.uniform(self.noise_variance[0], self.noise_variance[1])
            img = img + np.random.normal(0.0, variance, size=img.shape)
        return img, seg, box_coord, box_class


class BrightnessMultiplicativeTransform(object):
    """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param p_per_sample:
    """
    def __init__(self, multiplier_range=(0.5, 2), p_per_sample=1):

        self.p_per_sample = p_per_sample
        self.multiplier_range = multiplier_range

    def __call__(self, img, seg, box_coord, box_class):
        if np.random.uniform() < self.p_per_sample:
            multiplier = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1])
            img *= multiplier
        return img, seg, box_coord, box_class


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, p_per_sample=1):
        """
        Augments the contrast of data
        :param contrast_range: range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <1
        (in the inverval that was specified)
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def __call__(self, img, seg, box_coord, box_class):
        if np.random.uniform() < self.p_per_sample:
            mn = img.mean()
            if self.preserve_range:
                minm = img.min()
                maxm = img.max()
            if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                factor = np.random.uniform(self.contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
            img = (img - mn) * factor + mn
            if self.preserve_range:
                img[img < minm] = minm
                img[img > maxm] = maxm
        return img, seg, box_coord, box_class


class SimulateLowResolutionTransform(object):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis
        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:
        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), order_downsample=1, order_upsample=0, p_per_sample=1):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.p_per_sample = p_per_sample
        self.zoom_range = zoom_range

    def __call__(self, img, seg, box_coord, box_class):
        if np.random.uniform() < self.p_per_sample:

            shp = np.array(img.shape)

            if self.zoom_range[0] == self.zoom_range[1]:
                zoom = self.zoom_range[0]
            else:
                zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            target_shape = np.round(shp * zoom).astype(int)

            img_downsampled = resize(img.astype(float), target_shape, order=self.order_downsample, mode='edge', anti_aliasing=False)
            img = resize(img_downsampled, shp, order=self.order_upsample, mode='edge', anti_aliasing=False)
        return img, seg, box_coord, box_class


class RandomErasingTransform(object):
    """
    Augments the random erasing of data
    :param area_range: range of the size of crop patch, in the order of (x_min, y_min, z_min, x_max, y_max, z_max)
    :param p_per_sample:
    """
    def __init__(self, area_range=(8, 8, 4, 16, 16, 8), p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.area_range = area_range

    def __call__(self, img, seg, box_coord, box_class):
        if np.random.uniform() < self.p_per_sample:
            crop_size = [np.random.randint(low=self.area_range[i], high=self.area_range[i+3]) for i in range(3)]
            crop_center = [np.random.randint(low=crop_size[i]//2, high=img.shape[i] - crop_size[i]//2) for i in range(3)]
            img_e = img.copy()

            img_e[crop_center[0] - crop_size[0] // 2: crop_center[0] + crop_size[0] // 2,
                crop_center[1] - crop_size[1] // 2: crop_center[1] + crop_size[1] // 2,
                crop_center[2] - crop_size[2] // 2: crop_center[2] + crop_size[2] // 2] = img.mean()
            return img_e, seg, box_coord, box_class
        else:
            return img, seg, box_coord, box_class