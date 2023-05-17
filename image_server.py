#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import logging.config
import argparse

from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_restful import Api
from api import Stat, banner_cheking
from myutils.utils import save_half_keyword, save_keyword
from myutils.policy_checking import reload_file

import gdown

from werkzeug.utils import secure_filename

import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from telegram_notify import telegram_bot_sendtext

# app = Flask(__name__)
app = Flask(__name__, 
    static_url_path='/banner_detection/static', 
    static_folder='./static')    
api = Api(app)

UPLOAD_FOLDER = './static/uploads'
KEYWORD_FOLDER = './data'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])


my_module = banner_cheking()

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image_from_url(url, filename):
	output = f"./static/uploads/{filename}"
	gdown.download(url, output, quiet=False, verify=False)

@app.route('/banner_detection/html', methods=['GET','POST'])
def upload_check_ocr_html():
	start_time = time.time()
	if request.method.lower() == 'post':
		url = request.form.get("fname")
		if url is not None and url.startswith('http'):
			try:
				filename = "test_image.jpg"
				download_image_from_url(url, filename)
				r = my_module.predict_2(filename) 
				r['total_time'] = round(time.time()-start_time,5)
			except:
				return jsonify(dict(error=1,message="URL invaild"))
		else:	
			if 'file' not in request.files:			
				return jsonify(dict(error=1,message="Data invaild"))
			file = request.files['file']	
			if file.filename == '':			
				return jsonify(dict(error=1,message="Data invaild"))
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)			
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				r = my_module.predict_2(filename) 
				r['total_time'] = round(time.time()-start_time,5)
		item = {
		'text': None,
		'text_vietnamese': None,
		'time_detect_text': 0,
		'time_reg_eng': 0,
		'time_reg_vn': 0,
		'time_detect_image': 0,
		'Status': 0,  # 0: review, 1: keyword, 2: sexy, 3: crypto, 4: flag, 5: politician, 6: weapon
		'Reason': None,
		'total_time': 0,
		}
		dict_status = {
			0: 'review', 1: 'keyword', 2: 'sexy', 3: 'crypto', 4: 'flag', 5: 'politician', 6: 'weapon',
			7: 'atlas'
		}
		d = {
			'time_detect_text': 'Thời gian phát hiện vùng có text', 
			'time_reg_eng': 'Thời gian nhận dạng text theo tiếng Anh', 
			'text': 'Text theo tiếng Anh', 
			'time_reg_vn': 'Thời gian nhận dạng text theo tiếng Việt', 
			'text_vietnamese': 'Text theo tiếng Việt', 
			'time_detect_image': 'Thời gian chạy mô hình detect hình ảnh', 
			'Status': 'Trạng thái',  
			'Reason': 'Lý do',
			'total_time': 'Tổng thời gian',
			}
		for k,v in d.items():
			if k=='Status':
				flash(v+': '+str(dict_status[r[k]]))
			else:
				flash(v+': '+str(r[k]))
		
		#return jsonify(dict(error=0,data=r))
		new_filename = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
		new_filename2 = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'__.jpg'
		if os.path.isfile(os.path.join('./static/uploads', new_filename)):
			if not os.path.isfile(os.path.join('./static/uploads', new_filename)):
				return render_template('upload.html', filename=new_filename)
			else:
				return render_template('upload.html', filename=new_filename, filename2=new_filename2)
		else:
			return render_template('upload.html', filename=filename)

	return render_template('upload.html')

@app.route('/banner_detection/update_keyword', methods=['GET','POST'])
def update_keyword():
	if request.method.lower() == 'post':
		if 'file' not in request.files:			
			return jsonify(dict(error=1,message="Data invaild"))
		file = request.files['file']	
		if file.filename == '':			
			return jsonify(dict(error=1,message="Data invaild"))
		filename = secure_filename(file.filename)	
		if not filename.endswith('.xlsx'):
			return jsonify(dict(error=1,message="Data invaild"))
		try:
			path_file_excel = os.path.join(KEYWORD_FOLDER, 'Block QC 2022.xlsx')
			file.save(path_file_excel)
			save_half_keyword(path_excel=path_file_excel, name='ENG', name_save='halfban_eng')
			save_half_keyword(path_excel=path_file_excel, name='VI', name_save='halfban_vi')
			save_keyword(path_excel=path_file_excel, name_sheet='ENG', name='dic_eng')
			save_keyword(path_excel=path_file_excel, name_sheet='VI', name='dic_vi')
			save_keyword(path_excel=path_file_excel, name_sheet='vice_ENG', name='vice_eng')
			save_keyword(path_excel=path_file_excel, name_sheet='vice_VI', name='vice_vi')
			reload_file()
			os.remove(path_file_excel)
			return jsonify(dict(error=0, message="Update successful"))
		except:
			return jsonify(dict(error=1, message="Update failed"))

	return render_template('upload2.html')
	
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
			r = my_module.predict(filename) 
			return jsonify(dict(error=0,data=r))
	return render_template('upload.html')

@app.route('/banner_detection/check_banner', methods=['GET','POST'])
def check_banner():
	start_time = time.time()
	if request.method.lower() == 'post':	
		try:
			if 'file' not in request.files:			
				return jsonify(dict(error=1,message="Data invaild"))
			file = request.files['file']	
			if file.filename == '':			
				return jsonify(dict(error=1,message="Data invaild"))
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)			
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				r = my_module.predict_2(filename) 
				r['total_time'] = round(time.time()-start_time, 5)
				return jsonify(dict(error=0,data=r))
		except:
			# telegram_bot_sendtext(str("File error: " + file.filename))
			return jsonify(dict(error=1,message="Something Error"))
	return render_template('upload.html')

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
	app.debug = False
	app.run("0.0.0.0", port=port, threaded=True)

if __name__ == "__main__":
	main()
