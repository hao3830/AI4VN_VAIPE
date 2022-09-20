# cd yolor
cd yolov7
echo "RUNNING OCR..."
python3 run_ocr_on_pres.py
cd -

cd yolor
echo "RUNNING PILL DETECTION..."
python3 pill_infer.py --weights /data/weights/yolor_pill_det.pt --conf-thres 0.5 --iou-thres 0.45 --agnostic-nms
cd -

echo "RUNNING PILL CROPPING..."
python3 crop_pill.py

echo "RUNNING PILL RECOGNITION..."
cd FGVC-PIM
python3 inference_to_submit.py --c ./configs/vaipe_infer.yaml 
echo "DONE"


cd /workdir
cp /data/pill_det_results.csv /tmp

python3 /workdir/FGVC-PIM/post_process.py
