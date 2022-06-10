# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import cv2

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detection.predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

import os

path_root = './'
config_path = 'configs/ocr/'
model_path = 'models/'



def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(path_root, config_path, "icdar2013_101_FPN.yaml"))
    cfg.merge_from_list([])
    # Set model
    cfg.MODEL.WEIGHTS = os.path.join(path_root, model_path, "model_ic13_r101.pth")
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
    cfg.freeze()
    return cfg

def save_result_to_txt(prediction):

    classes = prediction['instances'].pred_classes
    boxes = prediction['instances'].pred_boxes.tensor

    result = {}
    id = 0
    for i in range(len(classes)):
        if classes[i]==0:
            xmin = int(boxes[i][0])
            ymin = int(boxes[i][1])
            xmax = int(boxes[i][2])
            ymax = int(boxes[i][3])

            result[id] = [xmin, ymin, xmax, ymax]
            id+=1
    return result

class DETECTION():
    def __init__(self) -> None:
        self.cfg = setup_cfg()
        self.model = None
        self.ok = False    # check if image has text or not

    def get_model(self):
        if self.model is not None:
            return self.model
        
        self.model = VisualizationDemo(self.cfg)
        print(">>>> load successfully textfusenet!!!!")
        return self.model
    
    def create_file_result(self, img_path, name):
        
        # output_path = os.path.join(path_root, save, name+'.json')
        dic_result = {}
        # try:
        # img_name = os.path.basename(img_path)
        print(">>> img path in detection:", img_path)
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(">>> img shape:", img.shape)
        self.model = self.get_model()
        prediction, vis_output, polygons = self.model.run_on_image(img)

        dic_bbox = save_result_to_txt(prediction)
        dic_result[name] = dic_bbox

        for k,v in dic_result[name].items():
            cv2.rectangle(img, (v[0], v[1]), (v[2], v[3]), (0, 0, 255), 1)
        cv2.imwrite(f'./static/uploads/{name}_.jpg', img)

        if len(dic_result[name]) == 0:
            self.ok = False
        else:
            self.ok = True

        return dic_result
