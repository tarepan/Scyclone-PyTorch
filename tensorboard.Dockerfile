# not yet supported in AWS Batch (https://tarepan.hatenablog.com/entry/2020/10/24/035916)
# Python==3.8.3 (checked by myself in Docker container)
# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

FROM python:3.9.0

RUN pip install boto3 tensorboard
# For pip install through git
# RUN apt-get update && apt-get install -y git

ENTRYPOINT ["tensorboard"]
# use `CMD` override for arguments.
#   c.f. ["--logdir", "DIRECTORY_PATH"]