from email.mime import image
import json
import h5py
import os
from os.path import dirname
import cv2
import numpy as np
import pycocotools
import random
import detectron2.data.transforms as T
import scipy.io as scio
from detectron2.structures import BoxMode
import argparse
import matplotlib.pyplot as plt


def get_particular_files(dir, end='depth.png'):
    all_files = os.listdir(dir)
    p_files = []
    for file in all_files:
        if file.endswith(end):
            p_files.append(file)
    return p_files


def bitmask2polygon(bitmask):
    # plt.imshow(bitmask.astype('uint8')*255)
    # plt.show()
    contours, hier = cv2.findContours(bitmask.astype(
        'uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_len_list = [len(x) for x in contours]
    cont_points = contours[np.argmax(contours_len_list)].squeeze()
    if len(cont_points) <= 4:
        return [], 0
    polygon = cv2.approxPolyDP(cont_points, 2.0, True).flatten().tolist()
    area = np.sum(bitmask.astype('uint8')).item()
    if len(polygon) <= 4:
        return [], 0
    return [polygon], area


def bitmsk2rle(bitmask):
    rle = pycocotools.mask.encode(np.asarray(bitmask, order="F"))
    rle['counts'] = rle['counts'].decode()
    return rle


def get_dicts(args):
    idx = 0
    # dataset_dicts = []
    depth_dir = os.path.join(os.getcwd(), args.name, args.phase, 'gray')
    seg_dir = os.path.join(os.getcwd(), args.name, args.phase, 'label')
    # meta_dir = os.path.join(os.getcwd(), args.name, 'meta')

    coco_anno = {"images": [],
                 "annotations": [],
                 "categories": [{'id': 0, 'name': '301'}]}
    files = os.listdir(depth_dir)
    for file in files:
        # record = {}
        ## occulusion ##
        # meta_path = os.path.join(meta_dir, file.split('.')[0]+'.mat')
        # meta = scio.loadmat(meta_path)
        print(file)
        # occulist = meta['oc'][0]
        depth_path = os.path.join(depth_dir, file)
        seg_path = os.path.join(seg_dir, file)
        # record['file_name'] = depth_path
        # record['image_id'] = idx
        a = cv2.imread(depth_path)
        height, width, _ = a.shape
        # record['height'] = height
        # record['width'] = width
        img_info = {
            'height': height,
            'width': width,
            'id': files.index(file),
            'file_name': file,
        }
        coco_anno["images"].append(img_info)
        # idx += 1
        seg_img = cv2.imread(seg_path)[:, :, 0]
        seg_ids = np.unique(seg_img)
        # seg_img = file['label'][:]
        # seg_ids = np.unique(seg_img)

        seg_ids = [id for id in seg_ids if id != 0]
        objs = []
        for id in seg_ids:
            # if id > len(occulist)-1:
            # continue
            # occ_r = occulist[id]
            bit_mask = (seg_img == id)
            # cv2.imshow('a', bit_mask.astype('uint8')*255)
            # cv2.waitKey(0)

            y_idxs, x_idxs = np.where(bit_mask == True)
            size_label = np.sum(bit_mask == True)

            # if (y_idxs.size != 0) and (x_idxs.size != 0) and (occ_r > args.oc):
            # rle = pycocotools.mask.encode(np.asarray(bit_mask, order="F"))
            # rle['counts'] = rle['counts'].decode()
            segment, area = bitmask2polygon(bitmask=bit_mask)

            # color, thickness and isClosed
            color = (255, 0, 0)
            thickness = 2
            isClosed = True
            # image = np.zeros_like(a)
            # # drawPolyline
            # segment = np.array(segment)
            # segment = segment.reshape((-1, 1, 2))
            # image = cv2.polylines(image, [segment], isClosed, color, thickness)
            # plt.imshow(image)
            # plt.show()
            # show image
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if area <= 900:
                continue
            anno = {
                "iscrowd": 0,
                "image_id": files.index(file),
                "bbox": [int(np.min(x_idxs)), int(np.min(y_idxs)), int(np.max(x_idxs))-int(np.min(x_idxs)), int(np.max(y_idxs))-int(np.min(y_idxs))],
                "segmentation": segment,
                "category_id": 0,
                "id": len(coco_anno['annotations']),
                "area": area
            }
            coco_anno["annotations"].append(anno)
        # dataset_dicts.append(record)

    return coco_anno


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train')
    parser.add_argument('--name', default='MY_NAME')
    parser.add_argument('--oc', default=0.9)
    args = parser.parse_args()
    out_dir = os.path.join(args.name, 'json')
    os.makedirs(out_dir, exist_ok=True)
    jname = args.phase + '.json'
    jdict = get_dicts(args)
    with open(out_dir+'/' + jname, 'w') as f:
        json.dump(jdict, f)
        print("json file: %s accomplished!" % jname)
