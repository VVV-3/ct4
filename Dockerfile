FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
RUN conda install -c conda-forge librosa