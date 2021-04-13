import os
import pickle
import numpy as np
import cv2
import random
import glob
import json
from pycocotools import _mask
from skimage.transform import matrix_transform
from cv2 import getRotationMatrix2D, warpAffine
import math
import threading
import scipy.io as scio
import argparse
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='09', required=False)
    parser.add_argument('--temp_file_type', type=str, default='png', required=False)
    parser.add_argument('--right', type=float, default=0.8, help='right margin of the effective range = right (ratio) * real width')
    parser.add_argument('--left', type=float, default=0.2)
    parser.add_argument('--upper', type=float, default=0.2)
    parser.add_argument('--bottom', type=float, default=0.8)
    parser.add_argument('--max_tem', type=int, default=50, help='maximum number of templates on a background')
    parser.add_argument('--min_tem', type=int, default=30, help='maximum number of templates on a background')
    parser.add_argument('--gen_num_per_base', type=int, default=10, help='maximum number of templates on a background')
    return parser.parse_args()

def prepare_template(args):
    json_paths = glob.glob(os.path.join(args.name, 'tm', '*.json'))
    tem_masks = []
    tem_dicts = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            full_dict = json.load(f)
        annos = full_dict['shapes']
        gray_path = os.path.join(args.name, 'tm', '%s.%s'%(json_path.split('/')[-1].split('.')[0], args.temp_file_type))
        gray = cv2.imread(gray_path)
        for an_dict in annos:
            poly_list = []
            for xy in an_dict['points']:
                poly_list.append(xy[0])
                poly_list.append(xy[1])
            rle = _mask.frPoly(poly=[poly_list], h=full_dict['imageHeight'], w=full_dict['imageWidth'])
            mask = _mask.decode(rle)[:,:,0]
            xs, ys = np.where(mask == True)
            # compute relative coordinates
            left_margin = np.min(xs)
            new_xs = xs - left_margin+1
            upper_margin = np.min(ys)
            new_ys = ys - upper_margin+1
            new_mask = np.zeros((903,1621))
            new_mask[new_xs,new_ys] = 1
            tem_masks.append(new_mask)
            dict = {}
            for x,y,nx,ny in zip(xs, ys, new_xs, new_ys):
                dict[(nx,ny)] = gray[x, y, :]
            tem_dicts.append(dict)
    return tem_masks, tem_dicts

def copy_pasteN_per_base(b,args):
    base_original = cv2.imread(os.path.join(args.name,'bg', b))
    height,width,c = base_original.shape
    RIGHT =  int(width * args.right)
    LEFT = int(width * args.left)
    BOTTOM = int(height * args.bottom)
    UPPER = int(height * args.upper)


    for i in range(args.gen_num_per_base):
        idx = 1
        sample_num = random.randint(args.min_tem, args.max_tem) # sample random N (a~b) templates
        base_img = base_original.copy()
        base_label = np.zeros((height,width,3))
        tem_addrs = {}
        occulusion_array = np.zeros(200)
        tem_num = len(tm_masks)
        enlarge_ratio = 5
        if enlarge_ratio * tem_num < args.max_tem:
            enlarge_ratio *= 2
        candidates = enlarge_ratio * list(range(tem_num))
        sample_temps = random.sample(candidates, sample_num)
        random.shuffle(sample_temps)
        for tem_idx in sample_temps:
            tmask = tm_masks[tem_idx]
            xs, ys = np.where(tmask == True)
            x_min, x_max, y_min, y_max = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
            left_margin = y_min -LEFT
            righ_margin = RIGHT - y_max
            top_margin = x_min - UPPER
            bottom_margin = BOTTOM - x_max
            tdict = tm_dicts[tem_idx]
            '''translate --> rotate--> paste'''
            tem_depth_image = np.zeros((height, width, 3)).astype('uint8')
            t_x = random.randrange(-top_margin, bottom_margin)
            t_y = random.randrange(-left_margin, righ_margin)
            tt = np.zeros((height, width))
            for x, y in zip(xs, ys):
                x1 = x + t_x
                y1 = y + t_y
                tt[x1, y1] = 1
                tem_depth_image[x1, y1, :] = tdict[(x, y)]
            tmask = (tt == 1)

            angle = random.randint(0, 36) * 10
            x1s, y1s = np.where(tmask == True)
            x1_min, x1_max, y1_min, y1_max = np.min(x1s), np.max(x1s), np.min(y1s), np.max(y1s)
            rot_mat = getRotationMatrix2D(center=((y1_min + y1_max) / 2, (x1_min + x1_max) / 2), angle=angle,
                                          scale=1)
            rotated_m = warpAffine(tmask.astype('uint8'), rot_mat, (width, height))
            rotated_d = warpAffine(tem_depth_image.astype('uint8'), rot_mat, (width, height))

            mask = (rotated_m == 1)
            base_img[mask] = rotated_d[mask]
            base_label[mask] = idx
            tem_addrs[idx] = tem_idx
            idx += 1

        for label_id, tem_id in tem_addrs.items():
            tem_mask = tm_masks[tem_id]
            tem_occ_mask = (base_label[:,:,0]==label_id)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(tem_mask.astype('uint8'))
            # plt.subplot(1,2,2)
            # plt.imshow(tem_occ_mask.astype('uint8'))
            # plt.show()
            occulusion_array[label_id] = (np.sum(tem_occ_mask)/np.sum(tem_mask))
        out_name = b.split('.')[0] + '_' + str(i)
        meta_dir = os.path.join(args.name, 'meta')
        os.makedirs(meta_dir, exist_ok=True)
        scio.savemat(os.path.join(meta_dir,out_name+'.mat'), {'oc':occulusion_array})
        gray_outdir = os.path.join(args.name,'train', 'gray')
        os.makedirs(gray_outdir, exist_ok=True)
        label_outdir = os.path.join(args.name, 'train', 'label')
        os.makedirs(label_outdir, exist_ok=True)
        cv2.imwrite(os.path.join(gray_outdir, out_name + '.png'), base_img)
        cv2.imwrite(os.path.join(label_outdir, out_name+ '.png'), base_label)
        print("%s has been saved." % (out_name))


if __name__ == "__main__":
    args = get_args()
    base_dir = os.path.join(args.name, 'bg' )
    tm_masks, tm_dicts = prepare_template(args)
    print('templates prepared!!')
    ths = [threading.Thread(target=copy_pasteN_per_base, args=(b,args,)) for b in os.listdir(base_dir)]
    for t in range(len(ths)):
        ths[t].start()
    # print(threading.activeCount())
