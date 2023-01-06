
def human_filter(w: int, h: int, lst: list, return_only_biggest_box: bool = True) -> any:
  result = []
  for dt in lst:
    cls, (x1,y1,x2,y2), score = dt
    if cls=='person' or cls==0:
      result.append([x1,y1,x2,y2])
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