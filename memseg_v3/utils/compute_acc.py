from xml.dom import INDEX_SIZE_ERR
import numpy as np
import mrcfile
from scipy.ndimage.measurements import label as lb
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

def convert_gt():
    img_path = "/Share/UserHome/huangweilin/Work/Tomo_data/proteasome/emd_7151/emd_7151_instance_mask.map"
    with mrcfile.open(img_path) as tomo:
        mask = tomo.data.copy()
    mask = np.expand_dims(mask, 0)
    bb_target = convert_seg_to_bounding_box_coordinates(mask)
    keep = ((bb_target[:,5] - bb_target[:,2]) < 50) * ((bb_target[:,4] - bb_target[:,1]) < 50) * ((bb_target[:,3] - bb_target[:,0]) < 50)
    bb_target = bb_target[keep, :]
    np.save("emd_7151.npy", bb_target)


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, z1, x2, y2, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]  # this is the gt box
        overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
    return overlaps

def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, z1, x2, y2, z2] (typically gt box)
    boxes: [boxes_count, (x1, y1, z1, x2, y2, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    z1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    y2 = np.minimum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou

def convert_seg_to_bounding_box_coordinates(img):

        '''
        This function generates bounding box annotations from given pixel-wise annotations.
        :param data_dict: Input data dictionary as returned by the batch generator.
        :param dim: Dimension in which the model operates (2 or 3).
        :param get_rois_from_seg: Flag specifying one of the following scenarios:
        1. A label map with individual ROIs identified by increasing label values, accompanied by a vector containing
        in each position the class target for the lesion with the corresponding label (set flag to False)
        2. A binary label map. There is only one foreground class and single lesions are not identified.
        All lesions have the same class target (foreground). In this case the Dataloader runs a Connected Component
        Labelling algorithm to create processable lesion - class target pairs on the fly (set flag to True).
        :param class_specific_seg_flag: if True, returns the pixelwise-annotations in class specific manner,
        e.g. a multi-class label map. If False, returns a binary annotation map (only foreground vs. background).
        :return: data_dict: same as input, with additional keys:
        - 'bb_target': bounding box coordinates (b, n_boxes, (y1, x1, y2, x2, (z1), (z2)))
        - 'roi_labels': corresponding class labels for each box (b, n_boxes, class_label)
        - 'roi_masks': corresponding binary segmentation mask for each lesion (box). Only used in Mask RCNN. (b, n_boxes, y, x, (z))
        - 'seg': now label map (see class_specific_seg_flag)
        '''

        for b in range(img.shape[0]):
            if np.sum(img[b]!=0) > 0:
                n_cands = int(np.max(img[b]))
                clusters = img[b]
                search_coords = np.argwhere(clusters > 0)
                coords_list = np.zeros([n_cands, 6])
                coords_list[:, :3] = 100000

                for idx in tqdm(range(search_coords.shape[0])):
                    z, y, x = search_coords[idx, :]
                    val = int(clusters[z, y, x]) - 1
                    coords_list[val, :3] = np.minimum(coords_list[val, :3], np.array([x, y, z]))
                    coords_list[val, 3:] = np.maximum(coords_list[val, 3:], np.array([x, y, z]))
                # for ii in tqdm(range(1, n_cands + 1)):
                #     roi = np.array((clusters == ii) * 1)  # separate clusters and concat
                #     seg_ixs = np.argwhere(roi != 0)
                #     z1 = seg_ixs[:, 0].min()
                #     z2 = seg_ixs[:, 0].max()
                #     y1 = seg_ixs[:, 1].min()
                #     y2 = seg_ixs[:, 1].max()
                #     x1 = seg_ixs[:, 2].min()
                #     x2 = seg_ixs[:, 2].max()
                #     if (x2 - x1) < 50 and (y2 - y1) < 50 and (z2 - z1) < 50:
                #         p_coords_list.append(np.array([x1, y1, z1, x2, y2, z2]))
                #     if ii > 10:
                #         break
        return coords_list

def nms_cpu(dets, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    z1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    z2 = dets[:, 5]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1) * (z2 - z1 + 1)
    scores = dets[:, 6]
    keep = []

    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        z11 = np.maximum(z1[i], z1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        z22 = np.minimum(z2[i], z2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        l = np.maximum(0, z22 - z11 + 1)  # the height of overlap

        overlaps = w * h * l
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep

def compute_pr(box_predict, box_gt):
    iou = 0.5
    nms_thre = 0.2
    size_bound = [5, 50]
    p_curve = []
    r_curve = []

    for i in tqdm(range(10)):
        score_thre = i * 0.02
        dets = []
        for idx in range(box_predict.shape[0]):
            if box_predict[idx, 6] < score_thre:
                continue
            x1, y1, z1, x2, y2, z2 = [int(c) for c in box_predict[idx, :6]]
            
            if (x2 - x1) > size_bound[1] or (x2 - x1) < size_bound[0]:
                continue
            if (y2 - y1) > size_bound[1] or (y2 - y1) < size_bound[0]:
                continue
            if (z2 - z1) > size_bound[1] or (z2 - z1) < size_bound[0]:
                continue

            dets.append(np.array([x1, y1, z1, x2, y2, z2, float(box_predict[idx, 6]), int(box_predict[idx, 7])]))

        if len(dets) is 0:
            r_curve.append(0.0)
            p_curve.append(1.0)
            continue
        dets = np.stack(dets)
        nms_keep = nms_cpu(dets, nms_thre)
        dets = dets[nms_keep, :]
        overlaps = compute_overlaps(dets, box_gt)
        recall = (overlaps.max(axis=0) > iou).mean()
        precision = (overlaps.max(axis=1) > iou).mean()
        r_curve.append(recall)
        p_curve.append(precision)
    return r_curve, p_curve

    # plt.plot(r_curve, p_curve)
    # plt.title("P-R Curve: IoU = {}".format(iou))
    # plt.show()
    # pdb.set_trace()
def plot_pr():
    # box_predict = np.load("/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/mwc_correct/proteasome.npy") 
    box_gt = np.load("./emd_7151.npy")

    box_full = np.load("/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/mwc_correct/emd_full_norm.npy")
    box_fpn = np.load("/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/mwc_correct/emd_fpn_norm.npy")
    box_no = np.load("/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/mwc_correct/emd.npy")
    #box_gt = np.load("/Share/UserHome/huangweilin/Work/deeptomo_project/deeptomo_v2/datasets/clean/proteasome_wm/ps7.04_res10.0_rseed6_ratio0.5_v0.44_ang-45_to_45/boundingbox_coords.npy")
    #box_gt = box_gt[:, [4, 2, 0, 5, 3, 1]]

    r_curve_full, p_curve_full = compute_pr(box_full, box_gt)
    r_curve_fpn, p_curve_fpn = compute_pr(box_fpn, box_gt)
    r_curve_no, p_curve_no = compute_pr(box_no, box_gt)

    plt.plot(r_curve_full, p_curve_full, 'r')
    plt.plot(r_curve_fpn, p_curve_fpn, 'b')
    plt.plot(r_curve_no, p_curve_no, 'g')
    plt.title("P-R Curve: IoU = {}".format(0.5))
    plt.legend(["Full BN", "FPN BN", "No BN"])
    plt.show()
    pdb.set_trace()




plot_pr()