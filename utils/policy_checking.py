import pandas as pd
import numpy as np
import json

dic_vi = json.load(open('./data/dic_vi.json'))
dic_eng = json.load(open('./data/dic_eng.json'))

def compare(key: str, text: str) -> bool:
    lst = text.split()
    k = key.split()

    for i in range(0, len(lst)-len(k)+1):
        tmp = lst[i:i+len(k)]
        if k==tmp:
            return True 
    return False

def check_text_vi(text: str) -> list:
    if text is None:
        return []
    low_text = text.lower()
    for key, lst_key in dic_vi.items():
        for word in lst_key:
            low_word = str(word).lower()
            if compare(low_word, low_text):
                return [key, word]
    
    return []

def check_text_eng(text: str) -> list:
    if text is None:
        return []
    low_text = text.lower()
    for key, lst_key in dic_eng.items():
        for word in lst_key:
            low_word = word.lower()
            if compare(low_word, low_text):
                return [key, word]
    
    return []


    
