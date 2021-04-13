from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
import argparse
import h5py
import os
import cv2
import numpy as np
import open3d as o3d
import pycocotools
import detectron2.data.transforms as T
import scipy.io as scio
import time
from PIL import Image

COLOR = [[0, 0.49038461538461536, 0.875, 0.8846153846153846], [1, 0.6538461538461539, 0.20192307692307693, 0.9134615384615384], [2, 0.8846153846153846, 0.8365384615384616, 0.3269230769230769], [3, 0.3076923076923077, 0.6153846153846154, 0.18269230769230768], [4, 0.8942307692307693, 0.3269230769230769, 0.14423076923076922], [5, 0.125, 0.4423076923076923, 0.875], [6, 0.46153846153846156, 0.3942307692307692, 0.5673076923076923], [7, 0.7692307692307693, 0.038461538461538464, 0.7115384615384616], [8, 0.5865384615384616, 0.125, 0.4326923076923077], [9, 0.19230769230769232, 0.5192307692307693, 0.8173076923076923], [10, 0.5769230769230769, 0.9230769230769231, 0.20192307692307693], [11, 0.08653846153846154, 0.10576923076923077, 0.6346153846153846], [12, 0.4519230769230769, 0.46153846153846156, 0.6153846153846154], [13, 0.9230769230769231, 0.27884615384615385, 0.19230769230769232], [14, 0.5096153846153846, 0.09615384615384616, 0.27884615384615385], [15, 0.16346153846153846, 0.375, 0.9230769230769231], [16, 0.5192307692307693, 0.7884615384615384, 0.3942307692307692], [17, 0.33653846153846156, 0.7019230769230769, 0.375], [18, 0.375, 0.6442307692307693, 0.21153846153846154], [19, 0.5576923076923077, 0.22115384615384615, 0.028846153846153848], [20, 0.38461538461538464, 0.5673076923076923, 0.8461538461538461], [21, 0.7884615384615384, 0.25, 0.7884615384615384], [22, 0.6346153846153846, 0.5384615384615384, 0.23076923076923078], [23, 0.5673076923076923, 0.2980769230769231, 0.125], [24, 0.22115384615384615, 0.23076923076923078, 0.5769230769230769], [25, 0.7115384615384616, 0.11538461538461539, 0.38461538461538464], [26, 0.028846153846153848, 0.38461538461538464, 0.28846153846153844], [27, 0.36538461538461536, 0.6057692307692307, 0.6538461538461539], [28, 0.07692307692307693, 0.5769230769230769, 0.1346153846153846], [29, 0.6730769230769231, 0.75, 0.5576923076923077], [30, 0.21153846153846154, 0.36538461538461536, 0.8942307692307693], [31, 0.6826923076923077, 0.8846153846153846, 0.8557692307692307], [32, 0.28846153846153844, 0.18269230769230768, 0.75], [33, 0.5480769230769231, 0.9134615384615384, 0.08653846153846154], [34, 0.40384615384615385, 0.07692307692307693, 0.7307692307692307], [35, 0.34615384615384615, 0.33653846153846156, 0.41346153846153844], [36, 0.9038461538461539, 0.25961538461538464, 0.49038461538461536], [37, 0.9423076923076923, 0.5288461538461539, 0.7980769230769231], [38, 0.2403846153846154, 0.057692307692307696, 0.33653846153846156], [39, 0.8653846153846154, 0.7980769230769231, 0.6923076923076923], [40, 0.25, 0.9038461538461539, 0.8269230769230769], [41, 0.4230769230769231, 0.21153846153846154, 0.36538461538461536], [42, 0.23076923076923078, 0.04807692307692308, 0.0673076923076923], [43, 0.8461538461538461, 0.34615384615384615, 0.22115384615384615], [44, 0.3557692307692308, 0.8076923076923077, 0.5192307692307693], [45, 0.41346153846153844, 0.028846153846153848, 0.2980769230769231], [46, 0.7980769230769231, 0.8653846153846154, 0.5], [47, 0.9134615384615384, 0.6730769230769231, 0.6826923076923077], [48, 0.47115384615384615, 0.7115384615384616, 0.25961538461538464], [49, 0.3269230769230769, 0.625, 0.40384615384615385], [50, 0.7403846153846154, 0.5576923076923077, 0.009615384615384616], [51, 0.5384615384615384, 0.41346153846153844, 0.7211538461538461], [52, 0.09615384615384616, 0.7596153846153846, 0.17307692307692307], [53, 0.9519230769230769, 0.6538461538461539, 0.34615384615384615], [54, 0.10576923076923077, 0.8557692307692307, 0.7403846153846154], [55, 0.4423076923076923, 0.6923076923076923, 0.9519230769230769], [56, 0.057692307692307696, 0.40384615384615385, 0.4519230769230769], [57, 0.3942307692307692, 0.14423076923076922, 0.8365384615384616], [58, 0.75, 0.49038461538461536, 0.6634615384615384], [59, 0.3173076923076923, 0.5961538461538461, 0.8653846153846154], [60, 0.6442307692307693, 0.08653846153846154, 0.5288461538461539], [61, 0.6057692307692307, 0.9326923076923077, 0.625], [62, 0.009615384615384616, 0.7307692307692307, 0.25], [63, 0.8365384615384616, 0.7403846153846154, 0.2403846153846154], [64, 0.25961538461538464, 0.17307692307692307, 0.4423076923076923], [65, 0.5961538461538461, 0.4807692307692308, 0.6730769230769231], [66, 0.019230769230769232, 0.8173076923076923, 0.3173076923076923], [67, 0.8557692307692307, 0.5, 0.8076923076923077], [68, 0.8173076923076923, 0.019230769230769232, 0.2692307692307692], [69, 0.6634615384615384, 0.6346153846153846, 0.3557692307692308], [70, 0.8269230769230769, 0.9423076923076923, 0.5961538461538461], [71, 0.04807692307692308, 0.19230769230769232, 0.16346153846153846], [72, 0.2980769230769231, 0.3173076923076923, 0.0], [73, 0.15384615384615385, 0.15384615384615385, 0.10576923076923077], [74, 0.4326923076923077, 0.5096153846153846, 0.04807692307692308], [75, 0.7211538461538461, 0.7692307692307693, 0.11538461538461539], [76, 0.8076923076923077, 0.1346153846153846, 0.5480769230769231], [77, 0.2692307692307692, 0.7211538461538461, 0.5096153846153846], [78, 0.4807692307692308, 0.28846153846153844, 0.7788461538461539], [79, 0.7019230769230769, 0.0, 0.5384615384615384], [80, 0.7788461538461539, 0.009615384615384616, 0.09615384615384616], [81, 0.7307692307692307, 0.16346153846153846, 0.4230769230769231], [82, 0.5288461538461539, 0.8461538461538461, 0.5865384615384616], [83, 0.6153846153846154, 0.4519230769230769, 0.019230769230769232], [84, 0.038461538461538464, 0.2692307692307692, 0.9326923076923077], [85, 0.9326923076923077, 0.8942307692307693, 0.038461538461538464], [86, 0.5, 0.5480769230769231, 0.057692307692307696], [87, 0.0673076923076923, 0.4326923076923077, 0.9423076923076923], [88, 0.27884615384615385, 0.6634615384615384, 0.6057692307692307], [89, 0.0, 0.5865384615384616, 0.7019230769230769], [90, 0.6923076923076923, 0.2403846153846154, 0.15384615384615385], [91, 0.20192307692307693, 0.9519230769230769, 0.07692307692307693], [92, 0.17307692307692307, 0.7788461538461539, 0.7692307692307693], [93, 0.875, 0.47115384615384615, 0.9038461538461539], [94, 0.18269230769230768, 0.4230769230769231, 0.4807692307692308], [95, 0.1346153846153846, 0.8269230769230769, 0.6442307692307693], [96, 0.11538461538461539, 0.3557692307692308, 0.47115384615384615], [97, 0.625, 0.0673076923076923, 0.3076923076923077], [98, 0.14423076923076922, 0.3076923076923077, 0.7596153846153846], [99, 0.7596153846153846, 0.6826923076923077, 0.46153846153846156]]
REFERRENCE = 15000

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='09mix_new_0013999.pth', type=str)
    parser.add_argument("--name", default='09', type=str)
    parser.add_argument("--score_thresh_test", default=0.8, type=float) # 0.5, 0.7, 0.9
    parser.add_argument("--nms_thresh_test", default=0.5, type=float, help='iou threshold when conducting NMS') #0.5, 0.8, 0.9
    parser.add_argument("--test_dir", default='./tt', type=str)  # test files have to follow certain file structure --depth --gray --ply (ply for depth scaling)
    parser.add_argument("--is_vis", default=True)
    parser.add_argument("--is_vis3d", default=False)   # only valid when is_vis = True
    parser.add_argument("--is_poseEst", default=False)
    parser.add_argument("--is_write_pose", default=False)
    # parser.add_argument("--vis3d", default=True)
    return parser

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(args.name, 'train_output','config.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(args.name, 'train_output', args.model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh_test
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thresh_test
    cfg.freeze()
    return cfg


# according to the picked mask only retain part of the point cloud
def picked_pcd(picked_mask, depth, intrinsics, origin_pc):
    scale = (np.max(origin_pc, axis=0) - np.min(origin_pc, axis=0))[2]
    bias = np.min(origin_pc, axis=0)[2]
    mask = np.logical_and((picked_mask == True), (depth > 0))
    xs, ys = np.where(mask)
    depth = depth[xs, ys].reshape(-1, 1)*scale + bias
    xs = xs.astype('float32')
    ys = ys.astype('float32')
    xys = np.vstack((ys, xs)).T
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]
    xys[:, 0] = (xys[:, 0] - cam_cx) / cam_fx
    xys[:, 1] = (xys[:, 1] - cam_cy) / cam_fy
    return np.hstack((xys * depth, depth))

def depth2pcd(depth, intrinsics, origin_pc, colors):
    scale = (np.max(origin_pc, axis=0) - np.min(origin_pc, axis=0))[2]
    bias = np.min(origin_pc, axis=0)[2]
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]
    non_zero_mask = (depth > 0)
    idxs = np.where(non_zero_mask)
    colors = colors[idxs[0], idxs[1], :].reshape(-1,3)
    z = depth[idxs[0], idxs[1]]*scale + bias
    x = (idxs[1]-cam_cx)*z/cam_fx
    y = (idxs[0]-cam_cy)*z/cam_fy
    pcd = np.stack((x, y, z), axis=1)

    return pcd, colors

def single_test_default_vis(cfg, img_path='/home/ds/Desktop/MyScripts/dataset/bmp_dst/val/0021_7.bmp'):
    img = cv2.imread(img_path, 0)
    height, width = img.shape[:2]
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    with torch.no_grad():
        img_tensor = torch.as_tensor(np.expand_dims(img, axis=2).astype("float32").transpose(2, 0, 1))
        inputs = {"image": img_tensor, "height": height, "width": width}
        outputs = model([inputs])[0]
        v = Visualizer(np.expand_dims(img, axis=2), metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('a', out.get_image())
        cv2.waitKey(0)

if __name__ == "__main__":
    intrinsics = np.identity(3, np.float32)
    intrinsics[0, 0] = 2.1780220711053257e+03
    intrinsics[1, 1] = 2.1780220711053257e+03
    intrinsics[0, 2] = 845.8888909163629
    intrinsics[1, 2] = 343.6686884068483

    args = argument_parser().parse_args()
    cfg = setup_cfg(args)
    output_dir = os.path.join(args.name, 'test_output', args.name + '_nms_' + str(args.nms_thresh_test) + 'score_'+str(args.score_thresh_test) + args.model_name.split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    test_save_dir = os.path.join(output_dir, args.test_dir)
    os.makedirs(test_save_dir)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()


    for img_path in os.listdir(args.test_dir):

        depth = cv2.cvtColor(np.asarray(Image.open(os.path.join(args.test_dir, img_path))), cv2.COLOR_RGB2BGR)
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth,2).repeat(3,2)

        com_name = img_path.split('.')[0]

        ## pred_mask ##
        height, width = depth.shape[:2]
        with torch.no_grad():
            input_tensor = torch.as_tensor(depth.astype("float32").transpose(2, 0, 1))
            inputs = {"image": input_tensor, "height": height, "width": width}
            outputs = model([inputs])[0]
            pred_masks = outputs['instances'].get('pred_masks').cpu().numpy()

            # pred_masks_idx = [i for i in range(len(pred_masks)) if (pred_masks[i].sum() > 2000)]
            # pred_masks = pred_masks[pred_masks_idx]

            if args.is_vis:
                scores = outputs['instances'].get('scores').cpu().numpy()
                img_3c = depth/255.0
                for color_id, mask in enumerate(pred_masks):
                    img_3c[mask == True] = COLOR[color_id][1:]
                    xs,ys = np.where(mask == True)
                    cv2.putText(img_3c, '%.2f'%(scores[color_id]), (int(np.median(ys)), int(np.median(xs))), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,0))
                img_3c = (img_3c * 255.0).astype(np.int32)

                cv2.imwrite(os.path.join(test_save_dir, com_name + '_2dvis.png'), img_3c)
                print('write %s'%(com_name))


