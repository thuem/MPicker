import numpy as np

def get_patch_crop_coords(scales, ratios, shape, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales_x, ratios_meshed = np.meshgrid(np.array(scales[0]), np.array(ratios))
    scales_y, _ = np.meshgrid(np.array(scales[1]), np.array(ratios))
    
    scales_x = scales_x.flatten()
    scales_y = scales_y.flatten()
    ratios_meshed = ratios_meshed.flatten()

    # Enumerate heights and widths from scales and ratios
    widths = scales_x / np.sqrt(ratios_meshed)
    heights = scales_y * np.sqrt(ratios_meshed)
    depths = np.tile(np.array(scales[2]), len(ratios_meshed))

    # Enumerate shifts in feature space
    shifts_x = np.arange(0, shape[0], anchor_stride[0])
    shifts_y = np.arange(0, shape[1], anchor_stride[1])
    shifts_z = np.arange(0, shape[2], anchor_stride[2])
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (x, y, z) and a list of (w, h, d)
    box_centers = np.stack(
        [box_centers_x, box_centers_y, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_widths, box_heights, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    
    return boxes