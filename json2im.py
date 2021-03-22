import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import logging
import logging.config
import yaml
from config import data_config
def visualize(lanes_vis, img, Id=range(4)):
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    img_vis = img.copy()
    for c, lane in enumerate(lanes_vis):
        print(len(lane))
        # print(lane)
        for j in range(len(lane)-1):
            cv2.line(img_vis, lane[j], lane[j+1], color[c], 16)
            cv2.circle(img_vis, lane[j+1], radius=5, color=(0, 255, 0))
    plt.imshow(img_vis)
    plt.show()

def lane_cmp(a, b):
    if all(i < 0 for i in a) or all(i < 0 for i in b):
        if all(i < 0 for i in b):
            return 1
        else:
            return -1
    else:
        a1 = [ i for i, x in enumerate(a) if x>0 ][0]
        a2 = [ i for i, x in reversed(list(enumerate(a))) if x>0 ][0]
        b1 = [ i for i, x in enumerate(b) if x>0 ][0]
        b2 = [ i for i, x in reversed(list(enumerate(b))) if x>0 ][0]
        start = max(a1, b1)
        end = min(a2, b2)
        if start > end:
            if a1 >= b1:
                a_x = a1
                b_x = b2
            else:
                a_x = a2
                b_x = b1
            if a_x >= b_x:
                return 1
            else:
                return -1
        else:
            a_x = a[start:end+1]
            a_mean = float(sum(a_x)) / len(a_x)
            b_x = b[start:end+1]
            b_mean = float(sum(b_x)) / len(b_x)
            if a_mean >= b_mean:
                return 1
            else:
                return -1

def lane_classify(lanes_vis):
    slope = []
    Id = []
    num = len(lanes_vis)
    for lane in lanes_vis:
        length = min(6, len(lane))
        k = (lane[-1][0] - lane[-length][0]) / float(lane[-1][1] - lane[-length][1])
        slope.append(k)
    print(slope)
    if num == 4:
        return range(4)
    if all(i > 0 for i in slope):
        assert num <= 2, "more than 2 right lanes!"
        Id = range(2, 2+num)
    elif all(i < 0 for i in slope):
        assert num <= 2, "more than 2 left lanes!"
        Id = range(2-num, 2)
    else:
        for i in range(num-1):
            if slope[i] <= 0 and slope[i+1] >= 0:
                assert i <= 3, "more than 3 left lanes?"
                Id = range(1-i, 1-i+num)
                break
            elif slope[i] > 0 and slope[i+1] < 0:
                raise ValueError('check data property!')
    return Id

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__=="__main__":
    CFG=data_config.data_cfg
    BASE_PATH=CFG.DATA.BASE
    TRAIN_PATH=os.path.join(BASE_PATH,CFG.DATA.TRAIN)
    OUTPUT=os.path.join(BASE_PATH,CFG.DATA.OUTPUT)
    print(TRAIN_PATH)
    JSONS=[os.path.join(TRAIN_PATH,json) for json in CFG.DATA.TRAIN_JSON]
    print(JSONS)


    with open('config/logging.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)

    for JSON in JSONS:
        json_train = [json.loads(line) for line in open(JSON).readlines()]
        for i, train in enumerate(json_train):
            # print(train)
            lanes = train['lanes']
            # lanes.sort(lane_cmp)
            y_samples = train['h_samples']
            raw_file = train['raw_file']
            # raw_file = raw_file.encode("ascii")
            img_file=os.path.join(TRAIN_PATH,raw_file)
            try:
                img = cv2.imread(img_file)
                # print(img)
                lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in lanes]
                lanes_vis_clean = [lane for lane in lanes_vis if len(lane) >= 2]
                Id = lane_classify(lanes_vis_clean)
                instance_img = np.zeros((720, 1280))
                binary_img = np.zeros((720, 1280))
                for j, lane in enumerate(lanes_vis_clean):
                    for k in range(len(lane)-1):
                        if Id[j] >= 0:
                            cv2.line(instance_img, lane[k], lane[k+1], (Id[j]+1)*50, 16)
                            cv2.line(binary_img, lane[k], lane[k+1],255, 16)
                
                gt_path = os.path.join(OUTPUT,'center' + raw_file[5:-3] + 'png')
                ensure_dir(gt_path)
                print(gt_path)
                # print(img)
                cv2.imwrite(gt_path, img)             
                
                gt_path = os.path.join(OUTPUT,'instance' + raw_file[5:-3] + 'png')
                ensure_dir(gt_path)
                cv2.imwrite(gt_path, instance_img)     

                gt_path = os.path.join(OUTPUT,'binary' + raw_file[5:-3] + 'png')
                ensure_dir(gt_path)
                cv2.imwrite(gt_path, binary_img)
                # visualize(lanes_vis_clean, img, Id)            
                # visualize(lanes_vis_clean, img, Id)
                logger.info(f"successful {raw_file}")
            except Exception:
                logger.error(f"failed to generate {os.path.join(TRAIN_PATH,raw_file)}")