FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

MAINTAINER Gökhan Karaahmetoğlu "365237@ogr.ktu.edu.tr"

RUN apt-get update && apt-get install -y wget vim unzip libgtk2.0-dev swig



RUN pip  install  matplotlib==3.1.0 


#change timezone
RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

WORKDIR /workspace


