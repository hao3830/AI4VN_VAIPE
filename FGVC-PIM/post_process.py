import json
from rapidfuzz.distance import Indel
from pathlib import Path
from PIL import Image,ImageOps,ImageFile
import magic
import re
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_pres_name(img_name):
    import json
    with open("/data/public_test/pill_pres_map.json") as f:
        data = json.load(f)

    for k, v in data.items():
        if img_name in v:
            return k + ".png"
    # n = img_name.split("_")[2]
    return ""


def get_similarity(text, _map):
    result = ""
    max_score = 0
    for key in _map:
        curr_score = Indel.normalized_similarity(text, key)
        if curr_score > max_score:
            max_score = curr_score
            result = key

    if max_score > 0.7:
        return _map[result]
    # print(text.encode('utf-8'))
    return None


def process_pres(pres, pres2pill):
    classes = []
    for ins in pres:
        text = ins["text"]
        if ") " in text and (" (" not in text):
            text = text[text.find(") "):].strip()
        text = text.replace(") ", "")
        text = text.lstrip("0123456789.- ")
        cls_ids = get_similarity(text, pres2pill)
        if not cls_ids:
            continue
        classes.extend(cls_ids)
    classes = list(filter(lambda x: x is not None, classes))
    return list(map(int, set(classes)))


def main():
    pill_img_dir = Path(
        "/data/public_test/pill/image")

    path_to_mapping_name_cls = "/workdir/FGVC-PIM/name_to_cls.json"
    name2cls = json.load(open(path_to_mapping_name_cls))

    path_to_ocr = Path("/data/ocr_results.json")
    ocr_results = json.load(path_to_ocr.open("r", encoding="utf-8"))

    out = open("/output/results.csv", "w")
    out.write("image_name,class_id,confidence_score,x_min,y_min,x_max,y_max\n")

    last_name = None
    count = {}

    with open("/output/_results.csv", "r") as f:
        data = f.readlines()
    data = data[1:]
    results = {}

    for idx, line in enumerate(data):
        img_name, cls1, cls2, cls3, prob1, prob2, prob3, xmin, ymin, xmax, ymax = line.strip().split(",")
        cls1 = int(cls1)
        cls2 = int(cls2)
        cls3 = int(cls3)
        prob1 = float(prob1)
        prob2 = float(prob2)
        prob3 = float(prob3)

        pres_ocr = ocr_results[get_pres_name(img_name)]
        pres_pills = process_pres(pres_ocr, name2cls)

        # t = tmp[img_name]

        cls_id = cls1
        prob = prob1

        if prob >= 0.999:
            cls_id = cls1
            prob = prob1
        elif cls1 in pres_pills:
            cls_id = cls1
            prob = prob1
        elif cls2 in pres_pills:
            cls_id = cls2
            prob = prob2
        elif cls3 in pres_pills:
            cls_id = cls3
            prob = prob3

        if prob < 0.5:
            prob += 0.5 

        if img_name not in results:
            results[img_name] = {}
        
        if cls_id not in results[img_name]:
            results[img_name][cls_id] =0
        results[img_name][cls_id] += 1

    last = None
    img_size = None

    for idx, line in enumerate(data):
        img_name, cls1, cls2, cls3, prob1, prob2, prob3, xmin, ymin, xmax, ymax = line.strip().split(",")
        cls1 = int(cls1)
        cls2 = int(cls2)
        cls3 = int(cls3)
        prob1 = float(prob1)
        prob2 = float(prob2)
        prob3 = float(prob3)
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        t = None
        if last is None or last != img_name:
            img_path = pill_img_dir / img_name
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            img_size = img.size

            # file_magic = magic.from_file(str(img_path))
            # regrex_result = re.findall("(\d+)x(\d+)", file_magic)
            # if len(regrex_result) > 1:
            #     if "upper-right" in file_magic:
            #         height, width = regrex_result[1]
            #     else:
            #         width, height = regrex_result[1]
            # else:
            #     if "upper-right" in file_magic:
            #         height, width = regrex_result[0]
            #     else:
            #         width, height = regrex_result[0]
            # img_size = (int(width), int(height))

            last = img_name

        thres = 5

        if ymin < thres:
            continue
        if ymax > img_size[1] - thres:
            continue
        if xmin < thres:
            continue
        if xmax > img_size[0] - thres:
            continue

        pres_ocr = ocr_results[get_pres_name(img_name)]
        pres_pills = process_pres(pres_ocr, name2cls)

        # t = tmp[img_name]

        cls_id = cls1
        prob = prob1

        if prob >= 0.999:
            cls_id = cls1
            prob = prob1
        elif cls1 in pres_pills:
            cls_id = cls1
            prob = prob1
        elif cls2 in pres_pills:
            cls_id = cls2
            prob = prob2
        elif cls3 in pres_pills:
            cls_id = cls3
            prob = prob3

        group_id = None  
        if  cls1 in results[img_name] and results[img_name][cls1] >= 2 and group_id not in results[img_name] :
            group_id = cls1
        elif  cls2 in results[img_name] and results[img_name][cls2] >= 2 and group_id not in results[img_name]:
            group_id = cls2
        elif  cls3 in results[img_name] and results[img_name][cls3] >= 2 and group_id not in results[img_name]:
            group_id = cls3
        if group_id is not None:
            cls_id = group_id 

        if 12 in pres_pills and cls1 in [98, 78, 5, 28]:
            cls_id = 12
            prob = prob1
            
        if prob < 0.5:
            prob += 0.5 

        if cls_id not in pres_pills:
            cls_id = 107
            prob = 1.0
        
        out.write("{},{},{},{},{},{},{}\n".format(
            img_name,
            cls_id,
            prob,
            xmin, ymin, xmax, ymax,
        ))

    out.close()



if __name__ == "__main__":
    main()
