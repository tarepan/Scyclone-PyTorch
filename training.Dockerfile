FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# latest s3fs needs Python=>3.7, so old verisons are needed for Python==3.6
RUN pip install s3fs==0.4.2
RUN pip install "fsspec[http, s3]==0.8.1"

RUN pip install git+https://github.com/tarepan/Scyclone-PyTorch

# temporal, for not-yet-existing checkpoint feature
RUN pip uninstall -y "pytorch-lightning"
RUN pip install "git+https://github.com/PyTorchLightning/pytorch-lightning.git@refs/pull/4402/head"

ENTRYPOINT ["python", "-m", "scyclonepytorch.Scyclone_main"]
# use `CMD` override for arguments.
#   c.f. [Understand how CMD and ENTRYPOINT interact](https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact)
