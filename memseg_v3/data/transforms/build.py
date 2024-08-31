# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_train_transforms(cfg, is_train=True):
    if not is_train:
        random_crop = False
    else:
        random_crop = cfg.TRANSFORM.RANDOM_CROP

    precrop_transform = T.PrecropTransform(crop_size=cfg.TRANSFORM.PRECROP_SIZE, random_crop=random_crop)
    spatial_transform = T.SpatialTransform(
        patch_size=cfg.INPUT.PATCH_SIZE, random_crop=random_crop,
        do_elastic_deform=cfg.TRANSFORM.DO_ROTATION, alpha=cfg.TRANSFORM.ELASTIC_ALPHA, 
        sigma=cfg.TRANSFORM.ELASTIC_SIGMA, p_el_per_sample=cfg.TRANSFORM.P_ELASTIC,
        do_rotation=cfg.TRANSFORM.DO_ROTATION, angle_x=cfg.TRANSFORM.ROTATION_X, 
        angle_y=cfg.TRANSFORM.ROTATION_Y, angle_z=cfg.TRANSFORM.ROTATION_Z, p_rot_per_sample=cfg.TRANSFORM.P_ROTATION,
        do_scale=cfg.TRANSFORM.DO_SCALE, scale=cfg.TRANSFORM.SCALE, p_scale_per_sample=cfg.TRANSFORM.P_SCALE,
        border_mode_data=cfg.TRANSFORM.IMG_BORDER_MODE, border_cval_data=cfg.TRANSFORM.IMG_BORDER_CVAL, order_data=cfg.TRANSFORM.IMG_BORDER_ORDER,
        border_mode_seg=cfg.TRANSFORM.SEG_BORDER_MODE, border_cval_seg=cfg.TRANSFORM.SEG_BORDER_CVAL, order_seg=cfg.TRANSFORM.SEG_BORDER_ORDER)

    noise_transform = T.GaussianNoiseTransform(noise_variance=cfg.TRANSFORM.NOISE_VAR, p_per_sample=cfg.TRANSFORM.P_NOISE)
    color_transform_brightness = T.BrightnessMultiplicativeTransform(multiplier_range=cfg.TRANSFORM.MULTIPLIER_RANGE, 
                                    p_per_sample=cfg.TRANSFORM.P_BRIGHTNES)
    color_tansform_contrast = T.ContrastAugmentationTransform(contrast_range=cfg.TRANSFORM.CONTRAST_RANGE, 
                                    preserve_range=cfg.TRANSFORM.PRESERVE_RANGE, p_per_sample=cfg.TRANSFORM.P_CONTRAST)
    
    resample_transform = T.SimulateLowResolutionTransform(zoom_range=cfg.TRANSFORM.ZOOM_RANGE, order_downsample=cfg.TRANSFORM.ORDER_DOWNSAMPLE, 
                                    order_upsample=cfg.TRANSFORM.ORDER_UPSAMPLE, p_per_sample=cfg.TRANSFORM.P_LOWRESOLUTION)
    # erase_transform = T.RandomErasingTransform(area_range=cfg.TRANSFORM.AREA_RANGE, p_per_sample=cfg.TRANSFORM.P_ERASE)

    transform = T.Compose(
        [
                precrop_transform,
                spatial_transform,
                noise_transform,
                color_transform_brightness,
                color_tansform_contrast,
                resample_transform
        ]
    )
    # if not cfg.CO_TRAIN:
    #     transform = T.Compose(
    #         [
    #             precrop_transform,
    #             spatial_transform,
    #             noise_transform,
    #             color_transform_brightness,
    #             color_tansform_contrast,
    #             resample_transform
    #         ]
    #     )
    # else:
    #     transform = T.Consistence(
    #         [
    #             precrop_transform,
    #             spatial_transform,
    #             resample_transform,
    #             erase_transform,
    #             noise_transform,
    #             color_transform_brightness,
    #             color_tansform_contrast,
    #         ], 2
    #     )
    
    return transform


def build_test_transforms(cfg): 
    transform = T.PrecropTransform(crop_size=cfg.INPUT.PATCH_SIZE, random_crop=False)
    return transform



