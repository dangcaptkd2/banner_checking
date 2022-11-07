import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from nsfw.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from nsfw.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from nsfw.yolov5.utils.plots import Annotator, colors, save_one_box
from nsfw.yolov5.utils.torch_utils import select_device, time_sync

import gc
import logging

class YOLOV5():
    def __init__(self, config) -> None:
        self.models_path = config["models"]
        self.thres = config["threshold"]
    
    def get_img_path(self, img_path):
        self.img_path = img_path
    @torch.no_grad()
    def run(self,
        weights=ROOT / 'model_human.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(320, 320),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_conf=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
        source = str(source)

        # Load model
        device = select_device(device)
        #device = torch.device('cpu')
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im, augment=augment)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    result = []
                    for *xyxy, conf, cls in reversed(det):
                        if True:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) #if save_conf else (cls, *xywh)  # label format
                            a = ('%g ' * len(line)).rstrip() % line
                            a = list(map(float, a.split()))
                            a[0] = names[int(a[0])]
                            result.append(a)
                            logging.getLogger('root').debug(f"Found object with confidence score: {round(float(conf), 3)}")
                else:
                    result=None


        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result        

    def get_human(self):
        logging.getLogger('root').debug("Running human model")
        return self.run(source=self.img_path, weights=self.models_path["detect_human"], conf_thres=self.thres["detect_human"])

    def get_flag(self):
        logging.getLogger('root').debug("Running flag model")
        return self.run(source=self.img_path, weights=self.models_path["detect_flag"], conf_thres=self.thres["detect_flag"])

    def get_weapon(self):
        logging.getLogger('root').debug("Running weapon model")
        return self.run(source=self.img_path, weights=self.models_path["detect_weapon"], conf_thres=self.thres["detect_weapon"])

    def get_crypto(self):
        logging.getLogger('root').debug("Running crypto model")
        return self.run(source=self.img_path, weights=self.models_path["detect_crypto"], conf_thres=self.thres["detect_crypto"])

    def get_boob(self, boob_img_path=None):
        logging.getLogger('root').debug("Running detect boob model")
        if boob_img_path is not None:
            a = boob_img_path
        else: 
            a = self.img_path
        return self.run(source=a, weights=self.models_path["detect_boob"], conf_thres=self.thres["detect_boob"])

