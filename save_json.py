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



def get_particular_files(dir, end = 'depth.png'):
    all_files = os.listdir(dir)
    p_files = []
    for file in all_files:
        if file.endswith(end):
            p_files.append(file)
    return p_files


def get_dicts(args):
    idx = 0
    dataset_dicts = []
    depth_dir = os.path.join(os.getcwd(), args.name, args.phase, 'gray')
    seg_dir = os.path.join(os.getcwd(), args.name, args.phase, 'label')
    meta_dir = os.path.join(os.getcwd(), args.name, 'meta')


    for file in os.listdir(depth_dir):
        record = {}
        ## occulusion ##
        meta_path = os.path.join(meta_dir, file.split('.')[0]+'.mat')
        meta = scio.loadmat(meta_path)
        print(file)
        occulist = meta['oc'][0]
        depth_path = os.path.join(depth_dir, file)
        seg_path = os.path.join(seg_dir, file)
        record['file_name'] = depth_path
        record['image_id'] = idx
        idx += 1
        a = cv2.imread(depth_path)
        height, width, _ = a.shape
        record['height'] = height
        record['width'] = width

        seg_img = cv2.imread(seg_path)[:,:,0]
        seg_ids = np.unique(seg_img)
        # seg_img = file['label'][:]
        # seg_ids = np.unique(seg_img)

        seg_ids = [id for id in seg_ids if id != 0]
        objs = []
        for id in seg_ids:
            if id > len(occulist)-1:
                continue
            occ_r = occulist[id]
            bit_mask = (seg_img == id)
            # cv2.imshow('a', bit_mask.astype('uint8')*255)
            # cv2.waitKey(0)
            y_idxs, x_idxs = np.where(bit_mask == True)
            size_label = np.sum(bit_mask == True)

            if (y_idxs.size != 0) and (x_idxs.size != 0) and (occ_r > args.oc):
                rle = pycocotools.mask.encode(np.asarray(bit_mask, order="F"))
                rle['counts'] = rle['counts'].decode()
                obj = {
                    "bbox": [int(np.min(x_idxs)), int(np.min(y_idxs)), int(np.max(x_idxs)), int(np.max(y_idxs))],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": rle,
                    "category_id": 0,
                    "occulusion": occ_r
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train')
    parser.add_argument('--name', default='09')
    parser.add_argument('--oc', default=0.9)
    args = parser.parse_args()
    out_dir = os.path.join(args.name, 'json')
    os.makedirs(out_dir, exist_ok=True)
    jname = args.phase + '.json'
    jdict = get_dicts(args)
    with open(out_dir+'/'+ jname,'w') as f:
        json.dump(jdict, f)
        print("json file: %s accomplished!"%jname)



