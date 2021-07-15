FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
LABEL maintainer="pengwei"
WORKDIR /idrlnet
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .