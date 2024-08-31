import numpy as np
from scipy.stats import norm
import pdb

def weighted_box_clustering(dets, thresh, n_ens):
    """
    consolidates overlapping predictions resulting from patch overlaps, test data augmentations and temporal ensembling.
    clusters predictions together with iou > thresh (like in NMS). Output score and coordinate for one cluster are the
    average weighted by individual patch center factors (how trustworthy is this candidate measured by how centered
    its position the patch is) and the size of the corresponding box.
    The number of expected predictions at a position is n_data_aug * n_temp_ens * n_overlaps_at_position
    (1 prediction per unique patch). Missing predictions at a cluster position are defined as the number of unique
    patches in the cluster, which did not contribute any predict any boxes.
    :param dets: (n_dets, (y1, x1, y2, x2, (z1), (z2), scores, box_pc_facts, box_n_ovs)
    :param thresh: threshold for iou_matching.
    :param n_ens: number of models, that are ensembled. (-> number of expected predicitions per position)
    :return: keep_scores: (n_keep)  new scores of boxes to be kept.
    :return: keep_coords: (n_keep, (y1, x1, y2, x2, (z1), (z2)) new coordinates of boxes to be kept.
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    z1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    z2 = dets[:, 5]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1) * (z2 - z1 + 1)

    scores = dets[:, 6]
    box_patch_id = dets[:, 7]
    box_pc_facts = dets[:, 8]
    box_n_ovs = dets[:, 9]

    # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
    order = scores.argsort()[::-1]
    
    keep_scores = []
    keep_coords = []

    while order.size > 0:
        i = order[0]  # higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy1 = np.maximum(y1[i], y1[order])
        yy2 = np.minimum(y2[i], y2[order])
        zz1 = np.maximum(z1[i], z1[order])
        zz2 = np.minimum(z2[i], z2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        d = np.maximum(0.0, zz2 - zz1 + 1)
        inter = w * h * d

        # overall between currently highest scoring box and all boxes.
        ovr = inter / (areas[i] + areas[order] - inter)

        # get all the predictions that match the current box to build one cluster.
        matches = np.argwhere(ovr > thresh)

        match_n_ovs = box_n_ovs[order[matches]]
        match_pc_facts = box_pc_facts[order[matches]]
        match_patch_id = box_patch_id[order[matches]]
        match_ov_facts = ovr[matches]
        match_areas = areas[order[matches]]
        match_scores = scores[order[matches]]

        # weight all socres in cluster by patch factors, and size.
        match_score_weights = match_ov_facts * match_areas * match_pc_facts
        match_scores *= match_score_weights

        # for the weigted average, scores have to be divided by the number of total expected preds at the position
        # of the current cluster. 1 Prediction per patch is expected. therefore, the number of ensembled models is
        # multiplied by the mean overlaps of  patches at this position (boxes of the cluster might partly be
        # in areas of different overlaps).
        n_expected_preds = n_ens * np.mean(match_n_ovs)

        # the number of missing predictions is obtained as the number of patches,
        # which did not contribute any prediction to the current cluster.
        n_missing_preds = np.max((0, n_expected_preds - np.unique(match_patch_id).shape[0]))

        # missing preds are given the mean weighting
        # (expected prediction is the mean over all predictions in cluster).
        denom = np.sum(match_score_weights) + n_missing_preds * np.mean(match_score_weights)

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / denom

        # compute weighted average of coordinates for the cluster. now only take existing
        # predictions into account.
        avg_coords = [np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(z1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(z2[order[matches]] * match_scores) / np.sum(match_scores)]

        # some clusters might have very low scores due to high amounts of missing predictions.
        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.01:
            keep_scores.append(avg_score)
            keep_coords.append(avg_coords)

        # get index of all elements that were not matched and discard all others.
        inds = np.where(ovr <= thresh)[0]
        order = order[inds]

    return keep_scores, keep_coords

def compute_box_factors(box_coords, patch_size):
    
    box_centers = np.stack([(box_coords[:, i] + box_coords[:, i + 3]) / 2 for i in range(3)]).T
    box_center_factors = np.mean(
                    norm.pdf(box_centers, loc=patch_size, scale=patch_size*0.8) * 
                    np.sqrt(2 * np.pi) * patch_size * 0.8, axis=1)
    
    box_overlap_factors = []
    for i in range(box_coords.shape[0]):
        box = box_coords[i, :]
        w = np.minimum(box_coords[:, 3] - box[0], box[3] - box_coords[:, 0])
        h = np.minimum(box_coords[:, 4] - box[1], box[4] - box_coords[:, 1])
        d = np.minimum(box_coords[:, 5] - box[2], box[5] - box_coords[:, 2])
        box_overlap_factors.append(sum((w > 0) * (h > 0) * (d > 0)) - 1)
    box_overlap_factors = np.array(box_overlap_factors)
    
    return box_center_factors, box_overlap_factors

