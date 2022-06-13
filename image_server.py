#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import logging.config
import yaml
import sys
import argparse

from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_restful import Api
from api import Stat, procssesing_image 

import sys

import os
import shutil
import gdown

from werkzeug.utils import secure_filename

import requests
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# app = Flask(__name__)
app = Flask(__name__, 
    static_url_path='/banner_detection/static', 
    static_folder='./static')    
api = Api(app)

UPLOAD_FOLDER = './static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image_from_url(url, filename):
	output = f"./static/uploads/{filename}"
	gdown.download(url, output, quiet=False)

@app.route('/banner_detection/html', methods=['GET','POST'])
def upload_check_oc_html():
	if request.method.lower() == 'post':
		start_time = time.time()	
		if 'file' not in request.files:			
			return jsonify(dict(error=1,message="Data invaild"))
		file = request.files['file']	
		if file.filename == '':			
			return jsonify(dict(error=1,message="Data invaild"))
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)			
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			r = procssesing_image(filename) 
			r['total_time'] = round(time.time()-start_time,5)
			d = {
					'time_detect_text': 'Thời gian phát hiện vùng có text (GPU)', 
					'time_reg_eng': 'Thời gian nhận dạng text theo tiếng Anh (GPU)', 
					'text': 'Text theo tiếng Anh', 
					'time_reg_vn': 'Thời gian nhận dạng text theo tiếng Việt (CPU)', 
					'time_reg_vn_in': 'Thời gian execution của model nhận dạng tiếng Việt', 
					'text_vietnamese': 'Text theo tiếng Việt', 
					'time_detect_image': 'Thời gian chạy mô hình detect hình ảnh (CPU)', 
					'status_sexy': 'Kết quả của mô hình detect sexy', 
					'flag': 'Kết quả của mô hình detect cờ', 
					'weapon': 'Kết quả của mô hình detect vũ khí',
					'status_face_reg': 'Kết của nhận diện chính khách',
					'ban keyword': 'Danh sách các từ khóa bị cấm', 
					'Status': 'Status', 
					'total_time': 'Tổng thời gian'
				}
			for k,v in d.items():
				flash(v+': '+str(r[k]))
			
			#return jsonify(dict(error=0,data=r))
			new_filename = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
			if os.path.isfile(os.path.join('./static/uploads', new_filename)):
				return render_template('upload.html', filename=new_filename)
			else:
				return render_template('upload.html', filename=filename)
	
	return render_template('upload.html')
	
@app.route('/banner_detection/check_ocr', methods=['GET','POST'])
def upload_check_ocr():
	if request.method.lower() == 'post':	
		if 'file' not in request.files:			
			return jsonify(dict(error=1,message="Data invaild"))
		file = request.files['file']	
		if file.filename == '':			
			return jsonify(dict(error=1,message="Data invaild"))
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)			
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			r = procssesing_image(filename) 
			return jsonify(dict(error=0,data=r))
	return render_template('upload.html')

@app.route('/banner_detection/check_ocr_url', methods =["GET", "POST"])
def upload_check_ocr_url():
	filename = "test_image.jpg"
	if request.method == "POST":
		url = request.form.get("fname")
		if allowed_file(url):
			download_image_from_url(url, filename)

			r = procssesing_image(filename=filename)
			return jsonify(dict(error=0,data=r))
		else:
			flash('Allowed only image url -> jpg, png, jpeg')
			return redirect(request.url)
	return render_template("upload.html")

@app.route('/banner_detection/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def main():
	api.add_resource(Stat, '/')
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', default=3050, help='port(default: 3050)')
	args = parser.parse_args()
	port = int(args.port)
	logging.info(f"Server start: {port}")
	app.debug = True
	app.run("0.0.0.0", port=port, threaded=True)

def test_full(filename):
	print(procssesing_image(filename))

if __name__ == "__main__":
	#filename = '21.png'
	#test_full(filename)
	main()
