import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import scipy.signal
from glob import glob
from tqdm import tqdm
import pdb

def plot_batch_proposals(imgs, boxes, batch_idx):
    imgs = np.array(imgs.cpu())
    boxes = np.array(boxes.cpu())
    batch_idx = np.array(batch_idx.cpu())

    for idx in range(imgs.shape[0]):
        img = imgs[idx, 0]
        box = boxes[np.nonzero(batch_idx == idx), :]

        col = img.max()

        for b in range(box.shape[1]):
            xmin = int(128 * box[0, b, 0])
            ymin = int(128 * box[0, b, 1])
            zmin = int(32 * box[0, b, 2])
            xmax = int(128 * box[0, b, 3])
            ymax = int(128 * box[0, b, 4])
            zmax = int(32 * box[0, b, 5])

            xmin, ymin, zmin = max(xmin, 0), max(ymin, 0), max(zmin, 0)
            xmax, ymax, zmax = min(xmax, 127), min(ymax, 127), min(zmax, 31)

            for i in range(xmin, xmax + 1):
                for j in range(ymin, ymax + 1):
                    if (j == ymax) or (j == ymin):
                        img[i, j, zmin:zmax] = col * 2
                    elif (i == xmax) or (i == xmin):
                        img[i, j, zmin:zmax] = col * 2
            
        with mrcfile.new('./output/proposals/{}.mrc'.format(idx), overwrite=True) as mrc:
            mrc.set_data(np.transpose(np.float32(img), [2, 0, 1]))

    return 

def plot_loss_curve(log_file):
    with open(log_file, 'r') as f:
        data = f.readlines()

    iteration = []
    loss = []

    for line in data:
        if line.find('Batch_time') >=0:
            line_info = line.split(' ')
            iteration.append(int(line_info[5][:-1]))
            loss.append(float(line_info[7][:-1]))
    iteration = np.array(iteration)
    loss = np.array(loss)
    return scipy.signal.savgol_filter(loss, 53, 3)

    
def nms_cpu(dets, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    z1 = dets[:, 4]
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

def plot_valid_box(img, box_coord, box_class, score, score_thre, size_bound, nms_thre):

    img = (img - img.min()) / (img.max()-img.min())
    shape = img.shape
    img_box = img.copy()
    dets = []

    for i in range(box_coord.shape[0]):

        if score[i] < score_thre:
            continue
        x1, y1, z1, x2, y2, z2 = [int(c) for c in box_coord[i, :]]
        x1, y1, z1 = max(x1, 0), max(y1, 0), max(z1, 0)
        x2, y2, z2 = min(x2, shape[0] - 1), min(y2, shape[1] - 1), min(z2, shape[2] - 1)
        
        if (x2 - x1) > size_bound[1] or (x2 - x1) < size_bound[0]:
            continue
        if (y2 - y1) > size_bound[1] or (y2 - y1) < size_bound[0]:
            continue
        if (z2 - z1) > size_bound[1] or (z2 - z1) < size_bound[0]:
            continue

        dets.append(np.array([x1, y1, x2, y2, z1, z2, float(score[i]), int(box_class[i])]))
    
    dets = np.stack(dets)
    nms_keep = nms_cpu(dets, nms_thre)
    dets = dets[nms_keep, :]


    for ind in range(dets.shape[0]):
        x1 = int(dets[ind, 0])
        y1 = int(dets[ind, 1])
        x2 = int(dets[ind, 2])
        y2 = int(dets[ind, 3])
        z1 = int(dets[ind, 4])
        z2 = int(dets[ind, 5])
        class_id = int(dets[ind, 7])

        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                if (j == y2) or (j == y1):
                    if class_id == 1:
                        img_box[i, j, z1: z2] = 1
                    else:
                        img_box[i, j, z1: z2] = 0
                elif (i == x2) or (i == x1):
                    if class_id == 1:
                        img_box[i, j, z1: z2] = 1
                    else:
                        img_box[i, j, z1: z2] = 0
    return img_box , dets


if __name__ == '__main__':
    # img_path = "/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/mwc_correct/proteasome.mrc"
    # with mrcfile.open(img_path, permissive=True) as tomo:
    #     img = tomo.data
    # box_path = "/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/mwc_correct/proteasome.npy"
    # box_info = np.load(box_path)

    # box_coord = box_info[:, :6]
    # box_score = box_info[:, 6]
    # box_class = box_info[:, 7]

    # img_box = plot_valid_box(np.transpose(img, [2, 1, 0]), box_coord, box_class, box_score, 0.2, [10, 40], 0.2)
    # with mrcfile.new('test.mrc', overwrite=True) as tomo:
    #     tomo.set_data(np.transpose(np.float32(img_box), [2, 1, 0]))
    
    # img_path_set = "/Share/UserHome/huangweilin/Work/Tomo_data/rubisco/c3_chloroplast_sta/*seg.mrc"
    # for img_path in tqdm(glob(img_path_set)):
    #     with mrcfile.open(img_path, permissive=True) as tomo:
    #         img = tomo.data.copy()

    #     box_path = img_path.replace("mrc", "npy")
    #     box_info = np.load(box_path)

    #     box_coord = box_info[:, :6]
    #     box_score = box_info[:, 6]
    #     box_class = np.ones(box_coord.shape[0])

    #     img_box, coords = plot_valid_box(np.transpose(img, [2, 1, 0]), box_coord, box_class, box_score, 0.05, [5, 50], 0.2)
    #     with mrcfile.new(img_path.replace(".mrc", "_box.mrc"), overwrite=True) as tomo:
    #         tomo.set_data(np.transpose(np.float32(img_box), [2, 1, 0]))
    #     x = (coords[:, 2] + coords[:, 0]) // 2
    #     y = (coords[:, 3] + coords[:, 1]) // 2
    #     z = (coords[:, 5] + coords[:, 4]) // 2
    #     score = coords[:, 6]

    #     np.savetxt(img_path.replace(".mrc", ".txt"), np.stack([x, y, z, score]).T, fmt="%.4f")
    loss_full_fpn = plot_loss_curve("/Share/UserHome/huangweilin/Work/MRF_seg/output/heinrich_dataset/full_data_fpn/log.txt")
    loss_partial_fpn = plot_loss_curve("/Share/UserHome/huangweilin/Work/MRF_seg/output/heinrich_dataset/partial_data_fpn/log.txt")
    loss_clean_fpn = plot_loss_curve("/Share/UserHome/huangweilin/Work/MRF_seg/output/heinrich_dataset/clean_data_fpn/log.txt")
    loss_full_unet = plot_loss_curve("/Share/UserHome/huangweilin/Work/MRF_seg/output/heinrich_dataset/full_data_unet/log.txt")
    loss_partial_unet = plot_loss_curve("/Share/UserHome/huangweilin/Work/MRF_seg/output/heinrich_dataset/partial_data_unet/log.txt")
    loss_clean_unet = plot_loss_curve("/Share/UserHome/huangweilin/Work/MRF_seg/output/heinrich_dataset/clean_data_unet/log.txt")
    plt.figure()
    plt.plot(loss_full_fpn)
    plt.plot(loss_partial_fpn)
    plt.plot(loss_clean_fpn)
    plt.plot(loss_full_unet)
    plt.plot(loss_partial_unet)
    plt.plot(loss_clean_unet)
    plt.legend(["loss_full_fpn", "loss_partial_fpn", "loss_clean_fpn", "loss_full_unet", "loss_partial_unet", "loss_clean_unet"])
    plt.show()
    pdb.set_trace()