import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.costom_logger import timeLogger
from data.dataset import build_loader

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models.builder import MODEL_GETTER

import cv2
from eval import my_evaluate_cm


def set_environment(args, tlogger):
    print("Setting Environment...")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    _, val_loader = build_loader(args)

    print("[Only Evaluation]")

    tlogger.print()

    ### = = = =  Model = = = =
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use our default setting
    print(model)


    #### Không tải model pretrained
    checkpoint = torch.load(args.pretrained, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model'])
    # start_epoch = checkpoint['epoch']


    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)
    tlogger.print()

    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """


    return val_loader, model

def main(args,tlogger):
    

    # result_df = result_df.drop(result_df[result_df.class_id == 107].index)
    val_loader, model = set_environment(args, tlogger)

    my_evaluate_cm(args, model, val_loader)

    # result_df.to_csv(
    #     "mobileNet_agu_107_prob.csv", index=False)


if __name__ == "__main__":
    tlogger = timeLogger()

    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    # args.device  = "cuda:7" if torch.cuda.is_available() else "cpu"
    # build_record_folder(args)

    # tlogger.print()

    main(args,tlogger)
