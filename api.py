import cv2
from flask_restful import Resource 

from detection import DETECTION
from recognition import RECOGNITION 
from recognition_vn.vietocr import RECOGNITION_VN
from utils.mid_process import mid_process_func_2, merge_boxes_to_line_text
from utils.policy_checking import check_text_eng, check_text_vi
from utils.utils import check_is_vn, clear_folder, save_image
from nsfw.nsfw import detect_flag, detect_weapon, detect_crypto, detect_nsfw

import torch

import os
import time
from collections import Counter

import torch
import time
import yaml
import gc

with open('./configs/common.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print("config:", config)

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

class banner_cheking():
    def __init__(self) -> None:
        self.path_image_root = config['path_save']['path_image_root']
        self.detect = DETECTION()
        self.recog = RECOGNITION()
        self.recog_vn = RECOGNITION_VN()
        
    def predict(self, filename: str) -> dict:
        item = {
            'text': None,
            'text_vietnamese': None,
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

        image_path = os.path.join(self.path_image_root, filename)
        name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')    
        since = time.time() 
        img = cv2.imread(image_path)
        if img is None:
            return item 

        since = time.time()
        result_nsfw = detect_nsfw(img_path=image_path, config=config)
        if isinstance(result_nsfw, str):
            item["status_face_reg"] = result_nsfw
            item['Status'] = 'Block'
            item['time_detect_image'] = round(time.time()-since, 5)
            return item
        elif result_nsfw:
            item["status_sexy"] =  True
            item['Status'] = 'Block'
            item['time_detect_image'] = round(time.time()-since, 5)
            return item
        if not config["test"]["test_sexy_only"]:
            item['flag'] = detect_flag(img_path=image_path, config=config)
            if item['flag']:
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                return item
            
            # item['weapon'] = detect_weapon(img_path=image_path, draw=True)
            # if item['weapon']:
            #     return item
            item['crypto'] = detect_crypto(img_path=image_path, config=config)
            if item['crypto']:
                item['Status'] = 'Block'
                item['time_detect_image'] = round(time.time()-since, 5)
                return item

        item['time_detect_image'] = round(time.time()-since, 5)

        if not config["test"]["test_sexy_only"]:
            # detect = DETECTION()
            result_detect = self.detect.create_file_result(img, name=name)
            # del detect
            torch.cuda.empty_cache()
            list_arr, sorted_cor = mid_process_func_2(image = img, result_detect=result_detect)
            item['time_detect_text'] = round(time.time()-since, 5)

            if len(list_arr)>0:
                # recog = RECOGNITION()
                since = time.time()
                result_eng = self.recog.predict_arr(bib_list=list_arr)
                # del recog
                torch.cuda.empty_cache()
                text_en = [result_eng[k][0] for k in result_eng if result_eng[k][1]>config["threshold"]["eng_text"]]

                if len(text_en)==0:
                    return item
                    
                item['text'] = ' '.join(text_en)
                item['time_reg_eng'] = round(time.time()-since, 5)

                if not check_is_vn(text_en, threshold=config["threshold"]["is_vn"]):
                    result_check_text_eng = check_text_eng(item['text'])
                    if result_check_text_eng:
                        item['Status'] = 'Block'
                        item['ban keyword'] = result_check_text_eng
                        return item

                else:
                    since = time.time()
                    bboxs = merge_boxes_to_line_text(img, sorted_cor)
                    # save_image(bboxs)
                    # recog_vn = RECOGNITION_VN()
                    text_vn = self.recog_vn.predict(bboxs, thres=config["threshold"]["vi_text"])
                    # del recog_vn
                    torch.cuda.empty_cache()
                    item['text_vietnamese'] = ' '.join(text_vn)
                    result_check_text_vi = check_text_vi(item['text_vietnamese'])
                    if result_check_text_vi:
                        item['Status'] = 'Block'
                        item['ban keyword'] = result_check_text_vi
                        item['time_reg_vn'] = round(time.time()-since, 5)
                        return item
                    item['time_reg_vn'] = round(time.time()-since, 5)
        clear_folder()
        return item

if __name__ == '__main__': 
    print("helllooooo")
    a = banner_cheking()
    r = a.predict('16.png')
    print(r)
