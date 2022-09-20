# from importlib.abc import Traversale
import matplotlib.pyplot as plt
import json
import argparse
import os
import cv2
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='MY_NAME/train/gray')
    parser.add_argument('--annotation', default='MY_NAME/json/train.json')
    parser.add_argument('--oc', default=0.9)
    args = parser.parse_args()
    with open(args.annotation, 'r') as f:
        annotations = json.load(f)

    annos = annotations['annotations']
    for anno in annos:
        img_path = os.path.join(args.image,annotations['images'][anno['image_id']]['file_name'])
        image = cv2.imread(img_path)
        segment = anno['segmentation']
        segment = np.array(segment)
        segment = segment.reshape((-1, 1, 2))
        image = cv2.polylines(image, [segment], isClosed=True, color=(255,0,0), thickness=2)
        plt.imshow(image)
        plt.show() 
        continue
