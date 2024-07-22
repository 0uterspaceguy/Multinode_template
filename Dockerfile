FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

