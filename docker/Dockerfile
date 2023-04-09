FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && apt-get upgrade --yes
RUN apt-get install vim emacs nano software-properties-common gnupg2 python3-pip --yes

WORKDIR /app

# Add extra tools for install
ADD installation_media .

RUN bash install_python_and_deps.sh

# Install additional requirements
# (Needed for deepspeed)
RUN apt-get install mpich --yes
RUN python -m pip install -r requirements.txt
RUN apt install git --yes
RUN python -m pip install --upgrade git+https://github.com/huggingface/transformers