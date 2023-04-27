import pandas as pd
import numpy as np
import json
from typing import Tuple

dic_vi = json.load(open('./data/dic_vi.json'))
dic_eng = json.load(open('./data/dic_eng.json'))
halfban_eng = json.load(open('./data/halfban_eng.json'))
vice_eng = json.load(open('./data/vice_eng.json'))
halfban_vi = json.load(open('./data/halfban_vi.json'))
vice_vi = json.load(open('./data/vice_vi.json'))

halfban_eng = [i.lower() for i in halfban_eng]

def compare(key: str, text: str) -> bool:
    lst = text.split()
    k = key.split()

    for i in range(0, len(lst)-len(k)+1):
        tmp = lst[i:i+len(k)]
        if k==tmp:
            return True 
    return False

def compare_special_value(key: list, text: str) -> bool:
    for word in key:
        if word not in text:
            return False 
    return True

def check_text(text: str, who: str) -> Tuple[bool, list]:
    if text is None:
        return []
    assert who != 'vietnamese' or who != 'english', 'who should be vietnamese or english'
    text = text.lower()
    if who == 'vietnamese':
        dic=dic_vi
        halfban = halfban_vi
        vice = vice_vi
    if who == 'english':
        dic=dic_eng
        halfban = halfban_eng
        vice = vice_eng
    tmp_list = []
    for key, lst_key in dic.items():
        if key=='simple':
            for special_value in lst_key:
                print(special_value)
                special_value = json.loads(special_value)
                if compare_special_value(key=special_value, text=text):
                    return True, [key, special_value]
        else:
            for word in lst_key:
                word = str(word).lower()
                if compare(word, text):
                    if word in halfban and len(vice[key])>0:
                        for vice_word in vice[key]:
                            vice_word = vice_word.lower()
                            if compare(vice_word, text) and vice_word!=word:
                                return True, [key, word, vice_word]
                        tmp_list = [key, word]
                    else:
                        return True, [key, word]
    
    return False, tmp_list

if __name__ == '__main__':
    text = "Lift saggy breasts with this meal plan"
    result = check_text(text, who='english')
    print(result)
    
