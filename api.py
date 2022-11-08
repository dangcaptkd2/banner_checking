import cv2
from flask_restful import Resource 

import logging
from logging.handlers import RotatingFileHandler

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')

logFile = './logs/log.log'

my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=50*1024, 
                                 backupCount=1, encoding=None, delay=0)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.DEBUG)

app_log = logging.getLogger('root')
app_log.setLevel(logging.DEBUG)

app_log.addHandler(my_handler)

from detection.textfusenet import DETECTION
from parseq.model import RECOGNITION
from recognition_vn.vietocr import RECOGNITION_VN
from utils.mid_process import mid_process_func_2, merge_boxes_to_line_text
from utils.policy_checking import check_text_eng, check_text_vi
from utils.utils import check_is_vn, clear_folder
from nsfw.nsfw import IMAGE_DETECT

import os
import time

import torch
import time
import yaml

with open('./configs/common.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print("config:", config)

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

class banner_cheking():
    def __init__(self) -> None:
        self.path_image_root = config['path_save']['path_image_root']
        self.IMAGE_DETECTION = IMAGE_DETECT(config=config)
        if config["run"]["ocr"]:
            self.detect:DETECTION = DETECTION(device=config["models"]["device"])
            self.recog:RECOGNITION = RECOGNITION(device=config["models"]["device"])
            self.recog_vn:RECOGNITION_VN = RECOGNITION_VN(device=config["models"]["device"])
        
    def predict(self, filename: str) -> dict:
        item = {
            'text': "",
            'text_vietnamese': "",
            'time_detect_text': 0,
            'time_reg_eng': 0,
            'time_reg_vn': 0,
            'status_sexy': False, 
            'status_face_reg': False,
            'flag': [],
            'weapon': [],
            'crypto': [],
            'time_detect_image': 0,
            'Status': 'Review',
            'total_time': 0,
            'ban keyword': [],
            'time_reg_vn_in': 0,
        }
        app_log.info(f"filename: {filename}")
        image_path = os.path.join(self.path_image_root, filename)
        name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')    
        since = time.time() 
        img = cv2.imread(image_path)
        if img is None:
            app_log.debug("Image is None -> Exit")
            return item 

        self.IMAGE_DETECTION.get_img_path(img_path=image_path)
        since = time.time()
        if config["run"]["sexy"]:
            result_nsfw = self.IMAGE_DETECTION.detect_nsfw()
            if isinstance(result_nsfw, str):
                item["status_face_reg"] = result_nsfw
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: Politician - {result_nsfw}")
                return item
            elif result_nsfw:
                item["status_sexy"] =  True
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: Sexy")
                return item
  
        if config["run"]["flag"]:
            item['flag'] = self.IMAGE_DETECTION.detect_flag()
            if item['flag']:
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: flag - {item['flag']}")
                return item
        
        if config["run"]["weapon"]:
            item['weapon'] = self.IMAGE_DETECTION.detect_weapon()
            if item['weapon']:
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: weapon")
                return item

        if config["run"]["crypto"]:
            item['crypto'] = self.IMAGE_DETECTION.detect_crypto()
            if item['crypto']:
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: crypto")
                return item

        item['time_detect_image'] = round(time.time()-since, 5)

        if config["run"]["ocr"]:
            # detect = DETECTION()
            result_detect = self.detect.create_file_result(img, name=name)
            # del detect
            torch.cuda.empty_cache()
            list_arr, sorted_cor = mid_process_func_2(image = img, result_detect=result_detect)
            item['time_detect_text'] = round(time.time()-since, 5)

            if len(list_arr)>0:
                app_log.info("The Image has text")
                since = time.time()
                result_eng = self.recog.predict(list_img=list_arr)
                item['text'] = result_eng
                item['time_reg_eng'] = round(time.time()-since, 5)
                app_log.info(f"Text: {result_eng}")

                if len(result_eng)==0:
                    return item
                
                if not check_is_vn(result_eng, threshold=config["threshold"]["is_vn"]):
                    app_log.info("Text is not Vietnamese")
                    result_check_text_eng = check_text_eng(item['text'])
                    if result_check_text_eng:
                        app_log.info(f"Contain ban keyword: {result_check_text_eng}")
                        item['Status'] = 'Block'
                        item['ban keyword'] = result_check_text_eng
                        return item

                else:
                    app_log.info("Text is Vietnamese")
                    since = time.time()
                    bboxs = merge_boxes_to_line_text(img, sorted_cor)
                    # save_image(bboxs)
                    # recog_vn = RECOGNITION_VN()
                    text_vn = self.recog_vn.predict(bboxs, thres=config["threshold"]["vi_text"])
                    # del recog_vn
                    torch.cuda.empty_cache()
                    item['text_vietnamese'] = ' '.join(text_vn)
                    app_log.info(f"Text vn: {item['text_vietnamese']}")
                    result_check_text_vi = check_text_vi(item['text_vietnamese'])
                    if result_check_text_vi:
                        app_log.info(f"Contain ban keyword: {result_check_text_vi}")
                        item['Status'] = 'Block'
                        item['ban keyword'] = result_check_text_vi
                        item['time_reg_vn'] = round(time.time()-since, 5)
                        return item
                    item['time_reg_vn'] = round(time.time()-since, 5)
        clear_folder()
        return item
    
    def predict_2(self, filename: str) -> dict:
        app_log.info(f"filename: {filename}")
        item = {
            'text': "",
            'text_vietnamese': "",
            'time_detect_text': 0,
            'time_reg_eng': 0,
            'time_reg_vn': 0,
            'time_detect_image': 0,
            'Status': 0,  # 0: review, 1: keyword, 2: sexy, 3: crypto, 4: flag, 5: politician, 6: weapon
            'Reason': "",
            'total_time': 0,
        }
        dict_result = {
            'review': 0,
            'keyword': 1,
            'sexy': 2,
            'crypto': 3,
            'flag': 4,
            'politician': 5,
            'weapon': 6,
        }

        image_path = os.path.join(self.path_image_root, filename)
        name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')    
        since = time.time() 
        img = cv2.imread(image_path)
        if img is None:
            app_log.debug("Image is None -> Exit")
            return item 
        self.IMAGE_DETECTION.get_img_path(img_path=image_path)
        since = time.time()
        if config["run"]["sexy"]:
            result_nsfw = self.IMAGE_DETECTION.detect_nsfw()
            if isinstance(result_nsfw, str):
                app_log.info(f"Status: Politician - {result_nsfw}")
                item["Reason"] = result_nsfw
                item['Status'] = dict_result['politician']
                item['time_detect_image'] = round(time.time()-since, 5)
                return item
            elif result_nsfw:
                app_log.info(f"Status: Sexy")
                item['Status'] = dict_result['sexy']
                item['time_detect_image'] = round(time.time()-since, 5)
                return item
  
        if config["run"]["flag"]:
            r = self.IMAGE_DETECTION.detect_flag()
            if r:
                app_log.info(f"Status: Sexy")
                item['Status'] = dict_result['flag']
                item['Reason'] = r
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: flag - {item['flag']}")
                return item
        
        if config["run"]["weapon"]:
            r = self.IMAGE_DETECTION.detect_weapon()
            if r:
                item['Status'] = dict_result['weapon']
                item['Reason'] = r
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: weapon")
                return item

        if config["run"]["crypto"]:
            r = self.IMAGE_DETECTION.detect_crypto()
            if r:
                item['Status'] = dict_result['crypto']
                item['Reason'] = r
                item['time_detect_image'] = round(time.time()-since, 5)
                app_log.info(f"Status: crypto")
                return item

        item['time_detect_image'] = round(time.time()-since, 5)

        if config["run"]["ocr"]:
            # detect = DETECTION()
            since = time.time()
            result_detect = self.detect.create_file_result(img, name=name)
            # del detect
            torch.cuda.empty_cache()
            list_arr, sorted_cor = mid_process_func_2(image = img, result_detect=result_detect)
            item['time_detect_text'] = round(time.time()-since, 5)
        
            if len(list_arr)>0:
                app_log.info("The Image has text")
                since = time.time()
                result_eng_2 = self.recog.predict(list_img=list_arr)
                item['text'] = result_eng_2
                item['time_reg_eng'] = round(time.time()-since, 5)
                app_log.info(f"Text: {result_eng_2}")


                if not check_is_vn(result_eng_2, threshold=config["threshold"]["is_vn"]):
                    app_log.info("Text is not Vietnamese")
                    result_check_text_eng = check_text_eng(item['text'])
                    if result_check_text_eng:
                        app_log.info(f"Contain ban keyword: {result_check_text_eng}")
                        item['Status'] = dict_result['keyword']
                        item['Reason'] = result_check_text_eng
                        return item

                else:
                    app_log.info("Text is Vietnamese")
                    since = time.time()
                    bboxs = merge_boxes_to_line_text(img, sorted_cor)
                    text_vn = self.recog_vn.predict(bboxs, thres=config["threshold"]["vi_text"])
                    torch.cuda.empty_cache()
                    item['text_vietnamese'] = ' '.join(text_vn)
                    app_log.info(f"Text vn: {item['text_vietnamese']}")
                    result_check_text_vi = check_text_vi(item['text_vietnamese'])
                    if result_check_text_vi:
                        app_log.info(f"Contain ban keyword: {result_check_text_vi}")
                        item['Status'] = dict_result['keyword']
                        item['Reason'] = result_check_text_vi
                        item['time_reg_vn'] = round(time.time()-since, 5)
                        return item
                    item['time_reg_vn'] = round(time.time()-since, 5)
        clear_folder()
        return item


if __name__ == '__main__': 
    print("helllooooo")
    a = banner_cheking()
    r = a.predict_2('acdysiggib.jpg')
    print(r)
