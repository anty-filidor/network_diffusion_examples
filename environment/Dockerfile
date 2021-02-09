FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt .

RUN pip install pip==9.0.1
RUN pip install --no-cache-dir -r requirements.txt
