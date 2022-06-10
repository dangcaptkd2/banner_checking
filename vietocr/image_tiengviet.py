#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import logging.config
import yaml
import sys
import argparse

from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_restful import Api
from api import Stat, processing
import sys

import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)
api = Api(app)

app.secret_key = "secret key"


@app.route('/viet_ocr/html', methods =["GET", "POST"])
def upload_check_ocr_url():
	if request.method == "POST":
		start_time = time.time()
		txt = request.form.get("fname")
		result = processing(txt)
		end_time = time.time()
		result['time_text_vn'] = round(end_time-start_time,5)
		print(">>>>>>>>> result:", result)
		return result
	return render_template("upload.html")

def main():
	api.add_resource(Stat, '/')
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', default=4050, help='port(default: 4050)')
	args = parser.parse_args()
	port = int(args.port)
	logging.info(f"Server start: {port}")
	app.debug = True
	app.run("0.0.0.0", port=port, threaded=True)

def test_full(name):
	print(processing(name))

if __name__ == "__main__":
	#filename = '21.png'
	#test_full(filename)
	main()
