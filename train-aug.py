import sys
import os
import json
import torch
import argparse
import detectron2.utils.comm as comm
import cv2
import random
import math

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch,HookBase
from detectron2 import model_zoo
from skimage.util import random_noise
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='09')
    parser.add_argument("--user_config_file", default="./user_config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--modelzoo_config_file", type=str, default=model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"), metavar="FILE", help="path to config file")
    # parser.add_argument("--is_erase", default=False)
    # parser.add_argument("--is_pepper_noise", default=False)
    # parser.add_argument("--is_gaussian_noise", default=False)

    parser.add_argument(
        "--resume",
        default= True,
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument('--num_gpus', type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser



def random_range_pepper(img, range=(0.0, 0.1)):
    amount = random.uniform(range[0], range[1])
    return random_noise(img, mode='pepper', amount=amount, clip=True)
def random_range_gaussian(img, range=(0.009, 0.01)):
    amount = random.uniform(range[0], range[1])
    return random_noise(img, mode='gaussian', clip=True, var=amount)

class RandomPepperNoise(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: (random_range_pepper(x, range=(0.0, 0.1))*255).astype('uint8'))

class RandomGaussianNoise(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: (random_range_gaussian(x, range=(0.0005, 0.01))*255).astype('uint8'))

class EraseTransform(T.ColorTransform):
    '''
    params:
    p: probabilty to conduct erasing
    scale: area compared with img area
    ratio: aspect ratio of the erase rectagle
    '''
    def __init__(self, p=0.5, scale=(0.001, 0.05), ratio=(0.1,5)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    def _get_params(self, img):
        img_h, img_w,_ = img.shape
        area = img_w * img_h
        for _ in range(10):
            erase_area = area * random.uniform(self.scale[0],self.scale[1])
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h<img_h and w<img_w):
                continue
            i = random.randint(0, img_h-h+1)
            j = random.randint(0, img_w-w+1)
            return i, j, h, w
        return 0, 0, img_h, img_w

    def apply_image(self, img):
        for i in range(10):
            if random.random() < self.p:
                i, j, h, w = self._get_params(img)
                img[i:i+h, j:j+w] = 0
        return img

class RandomErase(T.Augmentation):
    def get_transform(self, image):
        return EraseTransform(p=0.5, scale=(0.001, 0.008), ratio=(0.1,10))

class Trainer(DefaultTrainer):
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomBrightness(0.3, 2),
                T.RandomContrast(0.3, 2.5),
                T.RandomRotation([-45,45]),
                RandomGaussianNoise(),
                RandomPepperNoise()
   ]))

# class ValLossHook(HookBase):
#     def __init__(self, cfg):
#         # super().__init__()
#         self.cfg = cfg.clone()
#         # self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST   # no VAL
#         self._loader = iter(build_detection_train_loader(dataset=DatasetCatalog.get(self.cfg.DATASETS.TEST[0]),
#                                                          mapper=DatasetMapper(cfg, is_train=True,
#                                                                               augmentations=[
#                 T.RandomBrightness(0.6, 6.0),
#                 T.RandomContrast(0.3, 2.4),
#                 RandomGaussianNoise(),
#                 RandomPepperNoise(),
#                 T.RandomRotation([-45,45])
#                 # RandomErase()
#    ]
#                                                                               ),
#                                                          total_batch_size=cfg.SOLVER.IMS_PER_BATCH)) # param of build_test_loader
#
#     def after_step(self):
#         data = next(self._loader)
#         with torch.no_grad():
#             loss_dict = self.trainer.model(data)
#
#             losses = sum(loss_dict.values())
#             assert torch.isfinite(losses).all(), loss_dict
#
#             loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
#             losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#             if comm.is_main_process():
#                 self.trainer.storage.put_scalars(total_val_loss=losses_reduced, **loss_dict_reduced)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.modelzoo_config_file)
    cfg.merge_from_file(args.user_config_file)
    # cfg.MODEL.WEIGHTS = '/home/ds/Desktop/MyScripts/outfiles/indus2_depth200_ImageNet_lr0.0005/model_0000499.pth' # comment out if learn from scratch
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # comment out if learn from scratch
    cfg.OUTPUT_DIR = os.path.join(args.name, 'train_output')
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_dataset(cfg, data_dir):  # datadir must be dataset_large
    def get_box_dicts(json_path):
        # load json file to dict and change rle str to bytes
        with open(json_path, 'r') as f:
            j = json.load(f)
        for i in range(len(j)):
            for k in range(len(j[i]['annotations'])):
                j[i]['annotations'][k]['segmentation']['counts'] = j[i]['annotations'][k]['segmentation'][
                    'counts'].encode()
        return j
    DatasetCatalog.register(cfg.DATASETS.TRAIN[0],
                            lambda: get_box_dicts(os.path.join(data_dir,'json','train.json')))
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["box"])

    # DatasetCatalog.register(cfg.DATASETS.TEST[0],
    #                         lambda: get_box_dicts(os.path.join(data_dir,'json','val.json')))
    # MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=["box"])

def main(args):
    cfg = setup(args)
    register_dataset(cfg, data_dir=args.name)
    # trainer = DefaultTrainer(cfg)
    trainer = Trainer(cfg)
    # valhook = ValLossHook(cfg)
    # trainer.register_hooks([valhook])
    # trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    args = argument_parser().parse_args()
    launch(
        main(args),
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

