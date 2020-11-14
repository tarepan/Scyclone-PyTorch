# not yet supported in AWS Batch (https://tarepan.hatenablog.com/entry/2020/10/24/035916)
# Python==3.8.3 (checked by myself in Docker container)
# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Python==3.7.7 (checked by myself in Docker container)
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# For pip install through git
RUN apt install git

# latest s3fs needs Python=>3.7, so old verisons are needed for Python==3.6
# RUN pip install s3fs==0.4.2
# RUN pip install "fsspec[http, s3]==0.8.1"

RUN pip install git+https://github.com/tarepan/Scyclone-PyTorch

# temporal, for not-yet-existing checkpoint feature
RUN pip uninstall -y "pytorch-lightning"
RUN pip install "git+https://github.com/PyTorchLightning/pytorch-lightning.git@refs/pull/4402/head"

ENTRYPOINT ["python", "-m", "scyclonepytorch.Scyclone_main"]
# use `CMD` override for arguments.
#   c.f. [Understand how CMD and ENTRYPOINT interact](https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact)
