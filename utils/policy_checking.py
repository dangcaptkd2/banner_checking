import pandas as pd
import numpy as np
import json

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

def check_text_vi(text: str) -> list:
    if text is None:
        return []
    text = text.lower()
    for key, lst_key in dic_vi.items():
        for word in lst_key:
            word = str(word).lower()
            if compare(word, text):
                if word in halfban_vi and len(vice_vi[key])>0:
                    for vice_word in vice_vi[key]:
                        vice_word = vice_word.lower()
                        if compare(vice_word, text) and vice_word!=word:
                            return [key, word, vice_word]
                else:
                    return [key, word]
    
    return []

def check_text_eng(text: str) -> list:
    if text is None:
        return []
    text = text.lower()
    for key, lst_key in dic_eng.items():
        for word in lst_key:
            word = word.lower()
            if compare(word, text):
                if word in halfban_eng and len(vice_eng[key])>0:
                    for vice_word in vice_eng[key]:
                        vice_word = vice_word.lower()
                        if compare(vice_word, text) and vice_word!=word:
                            return [key, word, vice_word]
                else:
                    return [key, word]
    
    return []

if __name__ == '__main__':
    text = "Shiba Inu coin's value grew million percent. In other word even $1 investment in August 2020 would have generated $700K in profit. S100 OCT 2021 Sma Crypto 101 SmartNews Smarter news smarter moves Crypto news & investing basics Google Play Install"
    result = check_text_eng(text)
    print(result)
    
