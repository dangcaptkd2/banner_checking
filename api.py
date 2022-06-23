import cv2
from flask_restful import Resource 

from detection import DETECTION
from recognition import RECOGNITION 
from recognition_vn.vietocr import RECOGNITION_VN
from utils.mid_process import mid_process_func_2, action_merge, merge_boxes_to_line_text
from utils.policy_checking import check_text_eng, check_text_vi
from utils.utils import check_is_vn, clear_folder
from nsfw.nsfw import detect_flag, detect_weapon, detect_crypto, detect_nsfw

import torch

import os
import time
from collections import Counter

import torch
import time

import gc

path_models_classify = './models/'

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

class banner_cheking():
    def __init__(self) -> None:
        self.path_image_root = './static/uploads/'
        # self.detect = DETECTION()
        # self.recog = RECOGNITION()
        # self.recog_vn = RECOGNITION_VN()
        
    def predict(self, filename):
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
            'ban keyword': []
        }

        image_path = os.path.join(self.path_image_root, filename)
        result_nsfw = detect_nsfw(img_path=image_path, draw=True)
        if isinstance(result_nsfw, str):
            item["status_face_reg"] = result_nsfw
            item['Status'] = 'Block'
            return item
        elif result_nsfw:
            item["status_sexy"] =  True
            item['Status'] = 'Block'
            return item
        item['flag'] = detect_flag(img_path=image_path, draw=True)
        if item['flag']:
            item['Status'] = 'Block'
            return item
        
        # item['weapon'] = detect_weapon(img_path=image_path, draw=True)
        # if item['weapon']:
        #     return item
        item['crypto'] = detect_crypto(img_path=image_path, draw=True)
        if item['crypto']:
            return item

        name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        image_path = os.path.join(self.path_image_root, filename)     
        img = cv2.imread(image_path)

        detect = DETECTION()
        result_detect = detect.create_file_result(img, name=name)
        del detect
        torch.cuda.empty_cache()
        list_arr, sorted_cor = mid_process_func_2(image = img, result_detect=result_detect)
        
        if len(list_arr)>0:
            recog = RECOGNITION()
            result_eng = recog.predict_arr(bib_list=list_arr)
            del recog
            torch.cuda.empty_cache()
            text_en = [result_eng[k][0] for k in result_eng]
            item['text'] = ' '.join(text_en)

            if not check_is_vn(text_en):
                if check_text_eng(item['text']):
                    item['Status'] = 'Block'
                    return item

            else:
                bboxs = merge_boxes_to_line_text(img, sorted_cor)
                recog_vn = RECOGNITION_VN()
                text_vn = recog_vn.predict(bboxs)
                del recog_vn
                torch.cuda.empty_cache()
                item['text_vietnamese'] = ' '.join(text_vn)
                if check_text_vi(item['text_vietnamese']):
                    item['Status'] = 'Block'
                    return item

        clear_folder()
        return item

if __name__ == '__main__': 
    print("helllooooo")
    a = banner_cheking()
    r = a.predict('2.png')
    print(r)
