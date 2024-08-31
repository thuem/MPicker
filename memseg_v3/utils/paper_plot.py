import mrcfile
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import pdb

def nms_cpu(dets, scores, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    z1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    z2 = dets[:, 5]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1) * (z2 - z1 + 1)
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

def norm(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img * 255)
    return img

def img_fft(img):
    pad = np.float32(np.ones([(img.shape[1] - img.shape[0]) // 2, img.shape[1]]))
    img = np.concatenate([pad * img.mean(), img, pad * img.mean()])
    
    img = np.float32(img)
    img = np.fft.fftshift(np.fft.fft2(img))
    img = norm(np.log(np.abs(img)))
    return img

def plot_box(img, coord):
    size_bound = [10, 40]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for idx in range(coord.shape[0]):
        x1 = int(coord[idx, 0])
        y1 = int(coord[idx, 1])
        x2 = int(coord[idx, 2])
        y2 = int(coord[idx, 3])

        if (x2 - x1) > size_bound[1] or (x2 - x1) < size_bound[0]:
            continue
        if (y2 - y1) > size_bound[1] or (y2 - y1) < size_bound[0]:
            continue

        img[x1:x1+2, y1: y2 + 1, 0] = 0
        img[x1:x1+2, y1: y2 + 1, 1] = 255
        img[x1:x1+2, y1: y2 + 1, 2] = 0
        
        img[x2-1:x2+1, y1: y2 + 1, 0] = 0
        img[x2-1:x2+1, y1: y2 + 1, 1] = 255
        img[x2-1:x2+1, y1: y2 + 1, 2] = 0
        
        img[x1: x2 + 1, y1:y1+2, 0] = 0
        img[x1: x2 + 1, y1:y1+2, 1] = 255
        img[x1: x2 + 1, y1:y1+2, 2] = 0

        img[x1: x2 + 1, y2-1:y2+1, 0] = 0
        img[x1: x2 + 1, y2-1:y2+1, 1] = 255
        img[x1: x2 + 1, y2-1:y2+1, 2] = 0
    return img

def plot_valid_box(img, box_coord, box_class, score, score_thre, size_bound, nms_thre):

    img = (img - img.min()) / (img.max()-img.min())
    shape = img.shape
    img_box = img.copy()
    mask = np.zeros(img.shape)
    dets = []

    img_box[img_box < 0.8] = 0
    img_box[img_box > 0.8] = 1 

    for i in range(box_coord.shape[0]):

        if score[i] < score_thre:
            continue
        x1, y1, z1, x2, y2, z2 = [int(c) for c in box_coord[i, :]]
        x1, y1, z1 = max(x1, 0), max(y1, 0), max(z1, 0)
        x2, y2, z2 = min(x2, shape[2]), min(y2, shape[1]), min(z2, shape[0])
        
        if (x2 - x1) > size_bound[1] or (x2 - x1) < size_bound[0]:
            continue
        if (y2 - y1) > size_bound[1] or (y2 - y1) < size_bound[0]:
            continue
        if (z2 - z1) > size_bound[1] or (z2 - z1) < size_bound[0]:
            continue
        if z2 > shape[0] - 10 or z1 < 10:
            continue

        dets.append(np.array([x1, y1, z1, x2, y2, z2, float(score[i]), int(box_class[i])]))
    
    dets = np.stack(dets)
    nms_keep = nms_cpu(dets[:, :6], dets[:, 6], nms_thre)
    dets = dets[nms_keep, :]

    index = [i for i in range(dets.shape[0])]
    random.shuffle(index)
    dets = dets[index, :]

    for ind in range(dets.shape[0]):
        x1 = int(dets[ind, 0])
        y1 = int(dets[ind, 1])
        z1 = int(dets[ind, 2])
        x2 = int(dets[ind, 3])
        y2 = int(dets[ind, 4])
        z2 = int(dets[ind, 5])
        class_id = int(dets[ind, 7])

        for i in range(x1, x2):
            for j in range(y1, y2):
                if (j == y2 - 1) or (j == y1):
                    if class_id == 1:
                        img_box[z1: z2, j, i] = 1
                    else:
                        img_box[z1: z2, j, i] = 1
                elif (i == x2 - 1) or (i == x1):
                    if class_id == 1:
                        img_box[z1: z2, j, i] = 1
                    else:
                        img_box[z1: z2, j, i] = 1

    return img_box, dets

def filter_box(mask, coord, score, score_thre, size_bound, nms_thre):
    shape = mask.shape
    dets = []

    for i in range(coord.shape[0]):

        if score[i] < score_thre:
            continue
        x1, y1, z1, x2, y2, z2 = [int(c) for c in coord[i, :]]
        x1, y1, z1 = max(x1, 0), max(y1, 0), max(z1, 0)
        x2, y2, z2 = min(x2, shape[2]), min(y2, shape[1]), min(z2, shape[0])
        
        if (x2 - x1) > size_bound[1] or (x2 - x1) < size_bound[0]:
            continue
        if (y2 - y1) > size_bound[1] or (y2 - y1) < size_bound[0]:
            continue
        if (z2 - z1) > size_bound[1] or (z2 - z1) < size_bound[0]:
            continue
        if z2 > shape[0] - 2 or z1 < 2:
            continue

        dets.append(np.array([x1, y1, z1, x2, y2, z2, float(score[i]), 1]))
    
    dets = np.stack(dets)
    nms_keep = nms_cpu(dets[:, :6], dets[:, 6], nms_thre)
    dets = dets[nms_keep, :]

    coord = []

    for ind in range(dets.shape[0]):
        x1 = int(dets[ind, 0])
        y1 = int(dets[ind, 1])
        z1 = int(dets[ind, 2])
        x2 = int(dets[ind, 3])
        y2 = int(dets[ind, 4])
        z2 = int(dets[ind, 5])

        if mask[z1:z2, y1:y2, x1:x2].mean() > 0.1:
            coord.append(np.array([x1, y1, z1, x2, y2, z2]))
    coord = np.stack(coord)
    return coord

def plot_rubisco():

    img = mrcfile.open("./fig1/proteasome/mask.mrc", permissive=True).data.copy()
    img[:, :5, :] = 0
    img[:, -5:, :] = 0
    img[:, :, :5] = 0
    img[:, :, -5:] = 0
    box = np.load("./fig1/proteasome/detection_proteasome.npy")

    box_coord = box[:, :6]
    score = box[:, 6]
    box_class = box[:, 7]

    img_box = plot_valid_box(img, box_coord, box_class, score, score_thre=0.1, size_bound=[8, 40], nms_thre=0.1)

    with mrcfile.new("./fig1/GDH/mask_box.mrc", overwrite=True) as tomo:
        tomo.set_data(np.float32(img_box))

def plot_liposome():

    tomo = mrcfile.open("./fig1/liposome/liposome_seg.mrc").data.copy()
    tomo = tomo[33:70, 195:245, 390:440]

    with mrcfile.new("./fig1/liposome/liposome_seg_instance.mrc", overwrite=True) as f:
        f.set_data(tomo)
    pdb.set_trace()
    
    tomo[tomo > 0.5] = 1
    tomo[tomo <= 0.5] = 0

    z0 = 107
    y0 = 185
    x0 = 103

    tomo_xy = norm(tomo[z0, :, :])
    tomo_xz = norm(tomo[:, y0, :])
    tomo_yz = norm(tomo[:, :, x0])

    cv2.imwrite("./fig1/liposome/liposome_5um_5_bin4_seg_xy_box.jpg", tomo_xy)
    cv2.imwrite("./fig1/liposome/liposome_5um_5_bin4_seg_xz_box.jpg", tomo_xz)
    cv2.imwrite("./fig1/liposome/liposome_5um_5_bin4_seg_yz_box.jpg", tomo_yz)

def plot_gdh():
    
    img = -mrcfile.open("./fig1/GDH/emd_7141_crop.mrc").data.copy()
    mask = mrcfile.open("./fig1/GDH/mask.mrc").data.copy()
    coord_info = np.load("./fig1/GDH/detection_gdh.npy")

    mask[mask < 0.8] = 0
    mask[mask > 0.8] = 1 

    #img = mask

    z0 = 105
    y0 = 105
    x0 = 120
    coord = filter_box(mask, coord_info[:, :6], coord_info[:, 6], 0.03, [20,50], 0.1)

    tomo_xy = norm(img[z0, :, :])
    # coord_xy = coord[((coord[:, 2] + coord[:, 5]) / 2 < z0 + 15) * ((coord[:, 2] + coord[:, 5]) / 2 > z0 - 15), :]
    # coord_xy = coord_xy[:, [1, 0, 4, 3]]
    # tomo_xy = plot_box(tomo_xy, coord_xy)

    tomo_xz = norm(img[:, y0, :])
    # coord_xz = coord[((coord[:, 1] + coord[:, 4]) / 2 < y0 + 15) * ((coord[:, 1] + coord[:, 4]) / 2 > y0 - 15), :]
    # coord_xz = coord_xz[:, [2, 0, 5, 3]]
    # tomo_xz = plot_box(tomo_xz, coord_xz)

    tomo_yz = norm(img[:, :, x0])
    # coord_yz = coord[((coord[:, 0] + coord[:, 3]) / 2 < x0 + 15) * ((coord[:, 0] + coord[:, 3]) / 2 > x0 - 15), :]
    # coord_yz = coord_yz[:, [2, 1, 5, 4]]
    # tomo_yz = plot_box(tomo_yz, coord_yz)

    cv2.imwrite("./fig1/GDH/emd_7141_xy.jpg", tomo_xy)
    cv2.imwrite("./fig1/GDH/emd_7141_xz.jpg", tomo_xz)
    cv2.imwrite("./fig1/GDH/emd_7141_yz.jpg", tomo_yz)

def plot_proteasome():
    
    img = mrcfile.open("/Share/UserHome/huangweilin/Work/MoDL_3d/results/paper_exp/gdh_fda_k2.mrc").data.copy()
    mask = mrcfile.open("/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/paper_test/gdh_fda/gdh.mrc").data.copy()
    coord_info = np.load("/Share/UserHome/huangweilin/Work/Self-Supervised_DeepTomo/results/paper_test/gdh_fda/gdh.npy")

    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 1 

    z0 = 250
    y0 = 475
    x0 = 685

    mask, coord = plot_valid_box(mask, coord_info[:, :6], np.ones(coord_info[:, 6].shape), coord_info[:, 6], 0.06, [20,40], 0.2)

    with mrcfile.new("instance.mrc", overwrite=True) as tomo:
        tomo.set_data(np.float32(mask))

    gap = 10

    tomo_xy = norm(img[z0, :, :])
    coord_xy = coord[((coord[:, 2] + coord[:, 5]) / 2 < z0 + gap) * ((coord[:, 2] + coord[:, 5]) / 2 > z0 - gap), :]
    coord_xy = coord_xy[:, [1, 0, 4, 3]]
    tomo_xy = plot_box(tomo_xy, coord_xy)
    tomo_xy = tomo_xy[y0-150: y0+150, x0-150: x0+150]

    tomo_xz = norm(img[:, y0, :])
    coord_xz = coord[((coord[:, 1] + coord[:, 4]) / 2 < y0 + gap) * ((coord[:, 1] + coord[:, 4]) / 2 > y0 - gap), :]
    coord_xz = coord_xz[:, [2, 0, 5, 3]]
    tomo_xz = plot_box(tomo_xz, coord_xz)
    tomo_xz = tomo_xz[z0-100: z0+100, x0-150: x0+150]

    tomo_yz = norm(img[:, :, x0])
    coord_yz = coord[((coord[:, 0] + coord[:, 3]) / 2 < x0 + gap) * ((coord[:, 0] + coord[:, 3]) / 2 > x0 - gap), :]
    coord_yz = coord_yz[:, [2, 1, 5, 4]]
    tomo_yz = plot_box(tomo_yz, coord_yz)
    tomo_yz = tomo_yz[z0-100: z0+100, y0-150: y0+150]

    cv2.imwrite("./fig/gdh_xy.jpg", tomo_xy)
    cv2.imwrite("./fig/gdh_xz.jpg", tomo_xz)
    cv2.imwrite("./fig/gdh_yz.jpg", tomo_yz)


def plot_mwc():
    tomo_set = ["./fig1/proteasome/ps7.04_res25.0_rseed38_ratio0.5_v0.25_ang-45_to_45_recon_FDA0.2_pro.mrc",
                 "./fig1/proteasome/emd_7151_crop.mrc",
                 "./fig1/GDH/emd_7141_crop.mrc",
                 "./fig1/liposome/liposome_5um_5_bin4_crop.mrc"]
    tomo_slice = [228, 228, 105, 185]
    tomo_name = ['sim_pro', 'pro', 'gdh', 'liposome']
    ind = 0

    for tomo_path in tomo_set:
        with mrcfile.open(tomo_path, permissive=True) as tomo:
            img = -tomo.data
        with mrcfile.open(tomo_path.replace(".mrc", "_MWC.mrc"), permissive=True) as tomo:
            img_mwc = tomo.data
        
        img_zx = norm(img[:, tomo_slice[ind], :])
        img_zx_mwc = norm(img_mwc[:, tomo_slice[ind], :])

        img_zx_fft = img_fft(img_zx)
        img_zx_mwc_fft = img_fft(img_zx_mwc)

        # plt.imshow(img_zx_fft, cmap='gray')
        # plt.show()

        # plt.imshow(img_zx_mwc_fft, cmap='gray')
        # plt.show()


        cv2.imwrite("./fig3/{}.jpg".format(tomo_name[ind]), img_zx)
        cv2.imwrite("./fig3/{}_MWC.jpg".format(tomo_name[ind]), img_zx_mwc)
        cv2.imwrite("./fig3/{}_fft.jpg".format(tomo_name[ind]), img_zx_fft)
        cv2.imwrite("./fig3/{}_MWC_fft.jpg".format(tomo_name[ind]), img_zx_mwc_fft)
        ind += 1
    
plot_proteasome()

