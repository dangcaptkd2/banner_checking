FROM python:3.8-slim-buster
WORKDIR /app

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install g++5

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY . /app
ENTRYPOINT [ "python" ]
CMD [ "image_server.py" ]