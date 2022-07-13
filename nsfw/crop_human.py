from PIL import Image 
import cv2

def convert(dw, dh, dt):
  _, x, y, w, h = dt

  l = int((x - w / 2) * dw)
  r = int((x + w / 2) * dw)
  t = int((y - h / 2) * dh)
  b = int((y + h / 2) * dh)
  
  if l < 0:
      l = 0
  if r > dw - 1:
      r = dw - 1
  if t < 0:
      t = 0
  if b > dh - 1:
      b = dh - 1
  return _,l,r,t,b

def convert_filter(w: int, h: int, lst: list) -> list:
  result = []
  for dt in lst:
    cls, x1,x2,y1,y2 = convert(w, h, dt)
    result.append([x1,x2,y1,y2])
  return result

def human_filter(w: int, h: int, lst: list, return_only_biggest_box: bool = True) -> any:
  result = []
  for dt in lst:
    cls, x1,x2,y1,y2 = convert(w, h, dt)

    if cls=='person':
      result.append([x1,x2,y1,y2])
  if return_only_biggest_box:
    if len(result)>1:
      return find_largest_box(result)
    elif len(result)==1:
      return result[0]
    else:
      return []
  else:
    return result

def find_largest_box(result):
  largest = None
  biggest = 0
  for lst in result:
    x1,x2,y1,y2 = lst
    s = (x2-x1)*(y2-y1)
    if s > biggest:
      biggest = s
      largest = [x1,x2,y1,y2]
  
  return largest

# if __name__ == "__main__":
#     img_path = 'D:/sources/banner_checking/ocr/TextFuseNet/static/uploads/1.png'
#     img = cv2.imread(img_path)
#     h,w,_ = img.shape
    #result = human_filter(w,h,tmp)