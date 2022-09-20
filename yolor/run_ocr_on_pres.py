import argparse
import pandas as pd
import time
from pathlib import Path

from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from custom_utils import LoadImages, parse_yaml
from pathlib import Path
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T

from models.models import *
from utils.datasets import *
from utils.general import *


from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

import json


import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *


from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


class myYOLOR:
    def __init__(self, cfg, weights=None):
        if weights is None:
            weights = cfg.weights
        device = select_device(cfg.device)
        half = device.type != 'cpu'
        # Load model
        model = Darknet(cfg.cfg, cfg.imgsz).cuda()
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        self.cfg = cfg
        self.model = model
        self.half = half
        self.device = device

    def detect(self, img0):
        """
            img: numpy array
        """

        with torch.no_grad():
            img = letterbox(
                img0,
                new_shape=self.cfg.imgsz,
                auto_size=self.cfg.auto_size,
            )[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img, augment=self.cfg.augment)[0]

            pred = non_max_suppression(
                pred, self.cfg.conf_thres,
                self.cfg.iou_thres,
                classes=self.cfg.classes,
                agnostic=self.cfg.agnostic_nms,
            )

            boxes = []
            for i, det in enumerate(pred):
                if det is None or not len(det):
                    continue

                det[:, :4] = scale_coords(
                    img.shape[2:],
                    det[:, :4],
                    img0.shape,
                ).round()

                for *xyxy, conf, cls in det:
                    xmin, ymin, xmax, ymax = list(
                        map(lambda x: x.cpu().detach().numpy().astype(int).tolist(), xyxy))
                    boxes.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                        "score": conf.cpu().detach().item(),
                        "cls": cls.cpu().detach().item(),
                    })

            return boxes


def box2points(box):
    return [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
    ]


def get_gt(pres, box):
    box = box2points(box)
    for ins in pres:
        if "mapping" not in ins:
            continue
        _box = box2points(ins["box"])
        iou = calculate_iou(box, _box)
        if iou > 0.8:
            return ins["text"], ins["mapping"]

    return None, None


def main():
    cfg = parse_yaml("./cfg/yolor_ocr_infer.yaml")
    ROOT = Path(cfg.data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Cfg.load_config_from_name('vgg_seq2seq')

    # config['weights'] = 'transformerocr.pth'
    # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False

    ocr_model = Predictor(config)

    device = torch.device(device)

    model = myYOLOR(
        cfg=cfg,
        weights=cfg.weights,
    )

    results = {}

    with torch.no_grad():
        for img_path in tqdm(sorted(ROOT.iterdir())):
            # pres_path = LABEL_DIR / img_path.with_suffix(".json").name
            # pres = json.load(pres_path.open("r", encoding="utf-8"))

            pil_img = Image.open(img_path)

            result = model.detect(np.array(pil_img)[:, :, ::-1])

            texts = []
            for res in result:
                bb = list(map(lambda x: int(x+0.5), res["bbox"]))
                # bb[1] -= 1
                # bb[3] += 1
                # bb[0] -= 1
                # bb[2] += 1
                score = res["score"]
                # cls_id = res["cls"]
                cropped = pil_img.crop(bb)
                pred_text, prob = ocr_model.predict(cropped, return_prob=True)
                # print(text, prob)

                # text, clsId = get_gt(pres, bb)

                texts.append({
                    "text": pred_text,
                    "box": bb,
                    "box_score": score,
                    "ocr_score": prob,
                    # "gt_text": text,
                    # "mapping": clsId,
                })

            results[img_path.name] = texts

    json.dump(results, open(cfg.output_file, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
