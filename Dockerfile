FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

ENV TZ=Asia/Ho_Chi_Minh \
    DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 \
    CUDA_HOME="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing" \
    LANG=C.UTF-8

# Install all dependencies
RUN apt-get update && apt-get install wget libgbm-dev libgl1-mesa-glx libxrender1 libfontconfig1 -y
RUN apt-get install libglib2.0-0 -y
RUN pip install --upgrade pip
RUN pip install matplotlib==3.3.1 numpy==1.20.2 opencv-python==4.5.2.54 pillow==8.2.0 seaborn==0.11.0 scikit-learn rapidfuzz
RUN pip install protobuf==3.20.0 wandb torch==1.11.0 torchvision==0.12.0 timm==0.5.4 gdown Cython vietocr==0.3.8 easydict 
RUN apt-get install libmagic1 -y
RUN pip install python-magic

RUN mkdir -p /data/weights

WORKDIR /workdir

# SETUP CODE FOR YOLOR
WORKDIR /workdir/yolor
COPY ./yolor/requirements.txt .
RUN pip install -r requirements.txt

# SETUP CODE FOR YOLOv7
WORKDIR /workdir/yolov7
COPY ./yolov7/requirements.txt .
RUN pip install -r requirements.txt


COPY ./yolor /workdir/yolor
COPY ./yolov7 /workdir/yolov7
COPY ./FGVC-PIM /workdir/FGVC-PIM


# RUN gdown -O /data/weights/yolor_text_det.pt 16ZcVmpGUplcpywwb7MG_uxWBL9z2Ar5S
RUN gdown -O /data/weights/yolov7_text_det.pt 1-b0LsUPMXVrHvPAPRkT8rqvI77C2WPIe
RUN gdown -O /data/weights/yolor_pill_det.pt 1LUP0sM6NLGgJlj6Z4yH6qtEyUtvgfvlZ
# 300 epochs
# RUN gdown -O /data/weights/yolor_pill_det.pt 1HWBXY12YK4VvxVtAoejAJxLl3Fqr-42H
RUN gdown -O /data/weights/fgvc-pim-swinv2.pt 1vfLDh1aPg-TbPe-uqcDlpTFA_MZF72ic 


WORKDIR /workdir
COPY ./crop_pill.py .
COPY ./run.sh .

CMD ["bash", "./run.sh"]
