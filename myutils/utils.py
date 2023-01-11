import requests
from collections import Counter
import os
import shutil
import json
import cv2
import logging

def call_api_vi(name):
    API_ENDPOINT = "http://localhost:4050/viet_ocr/html"
    data = {
        'fname': (name)
    }
    r = requests.post(url = API_ENDPOINT, data = data)

    result = json.loads(r.text)
    print(">>>", result)
    return result

def call_api_nsfw(filename):
    API_ENDPOINT = "http://localhost:6050/nsfw/html"
    data = {
        'fname': (filename)
    }
    r = requests.post(url = API_ENDPOINT, data = data)

    print(">>>>>", r.text)
    result = json.loads(r.text)

    return result

def check_single_word(T: str) -> bool:
    """
    if T có khả năng là tiếng việt: return True
    else: return False
    """
    def check_english_1(T: str) -> bool:
      """
      Idea: trong tiếng việt, trừ từ 'xoong' ra thì không có từ nào có 2 chữ cái gần nhau mà giống nhau
      if not vietnamese: return True
      else: return False
      """
      if T=='xoong':
        return False
      for i in range(len(T)-1):
        if T[i]==T[i+1] and T[i].isalpha():
          return True
      return False
    vowel = 'ueoaiy'
    # 1.Chuyển chuỗi về không dấu, low case 
    T = T.strip().lower()

    # Nếu T có 2 chữ cái gần nhau mà giống nhau thì không phải là tiếng việt
    if check_english_1(T):
      return False

    # Nếu T là toàn là số thì là tiếng việt, nếu T chứa w,z,j,f thì không phải là tiếng việt
    if T.isnumeric():
      return True
    for i in ['w', 'z', 'j', 'f']:
      if i in T:
        return False

    # 2.Đánh dấu vị trí các nguyên âm : a, e, i, y, o, u 
    A = [i if T[i] in vowel else -1 for i in range(len(T))]
    
    # 4. Bỏ các phần từ không phải nguyên âm trong mảng A : loại các giá trị - 1.
    A = [i for i in A if i!=-1]
    # 3. Nếu từ không hề có bất kỳ nguyên âm nào  (mảng A toàn -1) -> từ không phải tiếng việt.
    if len(A)==0:
      return False

    # 5.Đếm trong chuỗi T các trường hợp có lớn hơn 1 nguyên âm cùng loại, ví dụ: aa, uu, oo,ii, yy
    new_T = [T[A[i]] for i in range(len(A))]
    C_5 = dict(Counter(new_T))
    for k,v in C_5.items():
      if v==2:
        """6. Nếu (5) = các case sau : "xoong" || "truu" || "tuu" || "khuyu" ||  "luu" || "cuu" || "suu"|| "buu"|| "muu"|| "huu"|| "nguu" || "gioi"
                || "giai"|| "giui" || "ngoeo"||"quau"|| "uou" thì xác định là tiếng Việt, ngược lại nếu (5) # tập trên sẽ là không phải tiếng Việt. """
        tmp = ["xoong", "truu", "tuu", "khuyu",  "luu", "cuu", "suu", "buu", "muu", "huu", "nguu", "gioi", "giai", "giui", "ngoeo", "quau", "uou"]
        for i in tmp:
          if i in T:
            return True
        return False
    
    # 7.Nếu mảng A có length > 1:
    if len(A)>1:
      # 7.1. mảng A có length > 3 : Trả ra là không phải tiếng Việt
      if len(A)>3:
        return False
      """
      7.2. Mảng A có length = 3: Nếu T có chứa cụm từ sau : uoi|uoc|uye|uya|oai|quai|giay|giau|giao|quao|giua|giuong|uay|oay|uan|yeu|ieu|queo 
      => là tiếng Việt. Nếu không trả ra là khác tiếng Việt.
      """
      if len(A)==3:
        tmp = ["uoi","uoc","uye","uya","oai","quai","giay","giau","giao","quao","giua","giuong","uay","oay","uan","yeu","ieu","queo"]
        for i in tmp:
          if i in T:
            return True
        return False
      """
      7.3. Trường hợp Mảng A có length < 3: Nếu có chứa cụm từ sau : ai|ay|ao|au|eo|eu|ia|ie|iu|oa|oe|oi|ua|ue|ui|uy|uo|ye|gio
      => là tiếng Việt. Nếu không trả ra là khác tiếng Việt.

      """
      if len(A)==2:
        tmp = ["ai", "ay", "ao", "au", "eo", "eu", "ia", "ie", "iu", "oa", "oe", "oi", "ua", "ue", "ui", "uy", "uo", "ye", "gio"]
        for i in tmp:
          if i in T:
            return True
        return False
    # 8.Mảng A có length =< 1
    if len(A)==1:
      #8.1.Nếu chuỗi T có length = 1 và chỉ nằm trong các chữ cái sau : a|e|o|u|i|y => là tiếng Việt
      if len(T)==1 and T in vowel:
        return True
      # 8.2.Trường hợp chuỗi T có chứa :  ac|at|am|an|ap|ec|et|em|en|ep|ic|it|im|in|ip|oc|ot|om|on|uc|ut|um|un|up|op => là tiếng Việt.  Nếu không trả ra là khác tiếng Việt.
      tmp = ["ac", "at", "am", "an", "ap", "ba", "ca", "da", "ga", "ha", "la", "ma", "na", "ra", "sa", "ta", "va", \
             "ec", "et", "em", "en", "ep", "be", "de", "he", "le", "me", "ne", "re", "se", "te", "ve", \
             "ic", "it", "im", "in", "ip", "bi", "di", "hi", "mi", "ni", "ri", "si", "ti", "vi", \
             "oc", "ot", "om", "on", "op", "bo", "co", "do", "go", "ho", "lo", "mo", "no", "ro", "so", "to", "vo",\
             "uc", "ut", "um", "un", "up", "bu", "cu", "du", "hu", "lu", "mu", "nu", "ru", "su", "tu", "vu"]
      for i in tmp:
        if i in T:
          return True
      return False

def check_is_vn(text: str, threshold=0.6) -> bool:
    lst = text.split()
    c=0
    for w in lst:
        if check_single_word(w)==1:
            c+=1
    score = c/len(lst)
    logging.getLogger('root').debug(f"Score vietnamese: {round(score, 3)}")
    if score>=threshold:
        return True 
    return False

def clear_folder() -> None:
    path_image_root = './static/uploads/'
    path_save_human4boob_detect = './tmp_images/human4boob_detect'
    
    if os.path.isdir(path_image_root):
      if len(os.listdir(path_image_root)) > 2000:
          logging.getLogger('root').info(f"Reset {path_image_root}")
          shutil.rmtree(path_image_root)
          os.makedirs(path_image_root)
          print("reset folder contain file!!!!!!")

    if os.path.isdir(path_save_human4boob_detect):
      if len(os.listdir(path_save_human4boob_detect)) > 1000:
        logging.getLogger('root').info(f"Reset {path_save_human4boob_detect}")
        shutil.rmtree(path_save_human4boob_detect)
        print("reset human4boob_detect contain file!!!!!!")

def save_image(lst: list, BGR: bool=True) -> None:
  path_save = './saved_image/'
  if not os.path.isdir(path_save):
    os.mkdir(path_save)
  
  for crop in lst:
    id = len(os.listdir(path_save))
    if BGR:
      cv2.imwrite(f'{path_save}{str(id)}.jpg', crop)
    else:
      cv2.imwrite(f'{path_save}{str(id)}.jpg', crop[:, :, ::-1])