import os
from PIL import Image
from zipfile import ZipFile
import imageio
from io import BytesIO
import cv2
import re
from tqdm import tqdm
import json

img_zip_path = 'D:/data/load_data/OCR_data/Training/imgs/digital.zip'
ann_zip_path = 'D:/data/load_data/OCR_data/Training/labels/digital.zip'

img_save_dir = 'D:/data/load_data/OCR_data/train_imgs'
os.makedirs(img_save_dir, exist_ok=True)
img_zipf = ZipFile(img_zip_path, 'r')
ann_zipf = ZipFile(ann_zip_path, 'r')
img_paths = [path for path in img_zipf.namelist() if path.endswith('jpg')]

gt_txt = open('D:/data/load_data/OCR_data/train_gt.txt', 'a', encoding='utf-8')
text_dict = {}
for img_path in tqdm(img_paths):
    ann_path = re.sub('.jpg', '.json', img_path)
    try:
        img_save_name = re.sub('/', '_', img_path)
        filename = img_save_name.split('.')[0]
        img = img_zipf.open(img_path)
        img = img.read()
        img = imageio.imread(BytesIO(img))
        img = Image.fromarray(img)
        json_file = ann_zipf.read(ann_path)
        with ann_zipf.open(ann_path) as json_file:
            anns = json.loads(json_file.read().decode('utf-8'))
        anns = anns['text']['word']
        i = 0
        for ann in anns:
            bbox = (xmin, ymin, xmax, ymax) = ann['wordbox']
            text = ann['value']
            try:
                text_dict[text]
            except KeyError:
                text_dict[text] = 1
                cropped_img = img.crop(bbox)
                cropped_img = cropped_img.resize((100, 32))
                img_save_path = os.path.join(img_save_dir, f'{filename}_{i}.jpg')
                i+=1
                cropped_img.save(img_save_path)
                label = f'imgs/{filename}_{i}.jpg\t{text}\n'
                gt_txt.write(label)
    except KeyError:
        pass

gt_txt.close()