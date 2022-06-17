import requests
import json
   
API_ENDPOINT = "https://rnd.fptonline.net/banner_detection/check_ocr"
#API_ENDPOINT = "http://localhost:3050/banner_detection/check_ocr"



def call(name: str, path: str) -> dict:
    ##### Truyền path ảnh vào đây
    #### format: 
    """
    data = {
        'file': ('tên_ảnh.jpg', open('path_ảnh.jpg', 'rb'))
    }
    """
    data = {
        'file': (name, open(path, 'rb'))
    }
    r = requests.post(url = API_ENDPOINT, files = data)

    print(">>>>>>>>", r.text)
    result = json.loads(r.text)
    # print(result)
    # print("Status:", result['data']['Status'])
    return result

# Lấy status của banner: print(result['data']['Status'])

# name = '21.png'
# path = 'C:/Users/quyennt72/Downloads/Banner_Block/'

# print(call(name, path+name))
path = 'C:/Users/quyennt72/Desktop/New folder/'
import os
c = 0
import time
start = time.time()
for name in os.listdir(path)[:100]:
    #print(name)
    result = call(name, path+name)
    
    c+=1
    print(c)
end = time.time()
ti = (end-start)/100
print(">>>>>>>>>>", ti)