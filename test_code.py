from tqdm import tqdm

with open('D:/data/OCR_DATASET/gt.txt', 'r', encoding='utf-8') as text_file:
    gt_list = text_file.readlines()

new_gt = open('D:/data/OCR_DATASET/new_gt.txt', 'a', encoding='utf-8')

for gt in tqdm(gt_list):
    path = gt.split('\t')[0]
    label = gt.split('\t')[1]
    if '.jpg' not in path:
        path = f'{path}.jpg'
    new_gt.write(f'{path}\t{label}')
    # if 'form' in path:
    #     filename = path.split('.')[0]
    #     remain = filename.split('_')
    #     num = str(int(remain[-1])-1)
    #     remain[-1] = num
    #     path = '_'.join(item for item in remain)
    # new_gt.write(f'{path}\t{label}')
new_gt.close()



