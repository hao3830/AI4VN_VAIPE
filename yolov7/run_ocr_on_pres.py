import argparse
import pandas as pd
import time
from pathlib import Path

from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
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


from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

import json


def draw_result(img, bbox, cls, conf, color, thickness):
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(
        bbox[2]), int(bbox[3])), color, thickness)
    return img


def load_model(weights, device, imgsz):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    return model, names, colors, stride, imgsz


def inference_draw(
    model,
    device,
    names,
    colors,
    thickness,
    process_img,
    imgsz,
    stride,
    augment,
    conf_thres,
    iou_thres,
    classes=None,
    agnostic_nms=False,
    half=False,
):
    dataset = LoadImages(process_img, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # print("pred", pred)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.cpu().tolist()

                for *xyxy, conf, cls_num in reversed(det_cpu):
                    bbox = xyxy
                    conf = float(conf)
                    cls = names[int(cls_num)]
                    process_img = draw_result(
                        process_img, bbox, cls, conf, colors[int(cls_num)], thickness)

    return process_img


def inference(
    model,
    device,
    names,
    process_img,
    imgsz,
    stride,
    augment,
    conf_thres,
    iou_thres,
    classes=None,
    agnostic_nms=False,
    half=False,
):
    dataset = LoadImages(process_img, img_size=imgsz, stride=stride)
    return_data = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # print("pred", pred)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.cpu().tolist()

                for *xyxy, conf, cls in reversed(det_cpu):
                    bbox = xyxy
                    conf = float(conf)
                    cls = names[int(cls)]

                    return_data.append(
                        {"bbox": bbox, "score": conf, "cls": cls})
    return return_data


# def calculate_iou(box_1, box_2):
#     poly_1 = Polygon(box_1)
#     poly_2 = Polygon(box_2)
#     iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#     return iou


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# def box2points(box):
#     return [
#         [box[0], box[1]],
#         [box[2], box[1]],
#         [box[2], box[3]],
#         [box[0], box[3]],
#     ]


def box2dict(box):
    return {
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
    }


def get_gt(pres, box):
    # box = box2points(box)
    box = box2dict(box)
    for ins in pres:
        if "mapping" not in ins:
            continue
        # _box = box2points(ins["box"])
        # iou = calculate_iou(box, _box)
        _box = box2dict(ins["box"])
        iou = get_iou(box, _box)
        if iou > 0.5:
            return ins["text"], ins["mapping"]

    return None, None


def main():
    cfg = parse_yaml("./cfg/yolov7_ocr_infer.yaml")
    ROOT = Path(cfg.data_dir)

    config = Cfg.load_config_from_name('vgg_seq2seq')

    # config['weights'] = './weights/transformerocr.pth'
    # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    # config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False

    ocr_model = Predictor(config)

    model, names, colors, stride, imgsz = load_model(
        cfg.weights,
        cfg.device,
        cfg.imgsz,
    )

    results = {}

    with torch.no_grad():
        for img_path in tqdm(sorted(ROOT.iterdir())):

            # pres_path = LABEL_DIR / img_path.with_suffix(".json").name
            # pres = json.load(pres_path.open("r", encoding="utf-8"))

            process_img = cv2.imread(str(img_path))
            pil_img = Image.open(img_path)

            result = inference(
                model,
                cfg.device,
                names,
                process_img,
                imgsz,
                stride,
                cfg.augment,
                cfg.conf_thres,
                cfg.iou_thres,
                cfg.classes,
                cfg.agnostic_nms,
                cfg.half,
            )

            texts = []
            for res in result:
                bb = list(map(int, res["bbox"]))
                bb[1] -= 5
                bb[3] += 5
                score = res["score"]
                # cls_id = res["cls"]
                cropped = pil_img.crop(bb)
                pred_text, prob = ocr_model.predict(cropped, return_prob=True)
                print(pred_text, prob)

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
