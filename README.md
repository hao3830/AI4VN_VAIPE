# Unstable - AIClub@UIT

## Quick start

Reproduce result on scoreboard with trained model

Build docker image:

```bash
docker build -t khiemledev/unstable_vaipe .
```

Run docker image:

```bash
docker run --rm --gpus <gpu_id> --shm-size 8G -v <your_output_path>:/output -v /data_folder:/data/public_test khiemledev/unstable_vaipe

# e.g
docker run -it --rm --gpus 1 --shm-size 8G -v $PWD/output1:/output -v /databases/VAIPE/public_test_new:/data/public_test khiemledev/unstable_vaipe
```

## Train

### Train text detection

We use [YOLOv7](https://github.com/WongKinYiu/yolov7.git) to detect pill name in prescription.

Steps to train:

1. Convert prescription data to YOLO format with single class (text only)
2. Cd into your yolov7 folder
3. Change your data path in `data/vaipe_text_det.yaml`
4. Run training command: `python3 train.py --weights yolov7-w6_training.pt --cfg cfg/training/yolov7-w6-text-det.yaml --data data/vaipe_text_det.yaml --hyp data/hyp.scratch.custom.yaml --epochs 100 --batch-size 4 --imgsz 1280`

### Train pill detector

We use [YOLOR](https://github.com/WongKinYiu/yolor.git) to detect pill in image.

Steps to train:

1. Convert pill data to YOLO format with single class (pill only) (all dataset for train and 15% for validation)
2. Cd into your yolor folder
3. Change your data path in `data/vaipe.yaml`
4. Dowload pretrained yolor_p6.pt from the repo
5. Run training command: `python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 16 --img 1280 1280 --data vaipe.yaml --cfg cfg/vaipe.cfg --weights 'yolor_p6.pt' --device 0,1 --sync-bn --name vaipe --hyp hyp.scratch.1280.yaml --epochs 19`

### Train pill classifier

We use [FGVC-PIM](git@github.com:chou141253/FGVC-PIM.git) as our classifier.

Steps to train:

1. Crop pill image from provided bounding box and convert it into ImageNet format (Only use class from 0 - 106).
2. We trained 2 phases, first, split 80% for train, 10% for validate and 10% for test (100 epochs). Then merge all folder train, validate, test in in train folder and keep old validate folder (100 epochs).
3. Cd into my FGVC-PIM
4. Change your data path and others configs in configs/vaipe.yaml
5. Run traning command: `python3 main.py --c configs/vaipe.yaml`
