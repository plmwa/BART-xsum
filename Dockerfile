FROM nvidia/cuda:11.7.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3-pip \ 
    sudo \
    wget \
    git \
    vim
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install torch torchvision \
    torchaudio jupyterlab transformers \
    torch pytorch-lightning hydra-core==1.0.4\
    wandb torchmetrics datasets nltk pandas

WORKDIR /work
#CMD ["/bin/bash"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]