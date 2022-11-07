import os 

require_folder = ['./tmp_images', './logs', './static', './static/uploads']

for folder in require_folder:
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print(f"Create folder: {folder}")