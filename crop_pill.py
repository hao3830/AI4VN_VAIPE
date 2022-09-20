from typing import Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from multiprocessing import Pool, Process


def crop_img(img_path: Path, boxes: Tuple, save_dir: Path):
    img = cv2.imread(str(img_path))
    for idx, bb in boxes:
        cropped = img[bb[1]:bb[3], bb[0]:bb[2]]

        save_path = save_dir / (f"{idx}${img_path.name}")
        cv2.imwrite(str(save_path), cropped)


def main():
    det_result_path = Path("/data/pill_det_results.csv")
    img_dir = Path("/data/public_test/pill/image")
    save_dir = Path("/data/det_cropped/0")
    save_dir.mkdir(parents=True, exist_ok=True)

    with det_result_path.open("r") as f:
        lines = f.readlines()
        lines = lines[1:]  # Remove heading

    imgs = {}

    for idx, line in tqdm(enumerate(lines)):
        line = line.strip()
        if line == "":
            continue
        line = line.split(",")

        if line[0] not in imgs:
            imgs[line[0]] = []

        imgs[line[0]].append((idx, list(map(int, line[-4:]))))

    with Pool(4) as p:
        p.starmap(crop_img, list([(img_dir / k, v, save_dir)
                  for k, v in imgs.items()]))


if __name__ == "__main__":
    main()
