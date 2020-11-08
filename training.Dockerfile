FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN pip install git+https://github.com/tarepan/Scyclone-PyTorch

ENTRYPOINT ["python", "-m", "scyclonepytorch.Scyclone_main"]
# use `CMD` override for arguments.
#   c.f. [Understand how CMD and ENTRYPOINT interact](https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact)
