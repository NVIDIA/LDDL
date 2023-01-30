ARG TAG
# Import a NGC PaddlePaddle container as the base image.
# For more information on NGC PaddlePaddle containers, please visit:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paddlepaddle
FROM nvcr.io/nvidia/paddlepaddle:${TAG}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update -qq && \
    apt-get install -y git vim tmux && \
    rm -rf /var/cache/apk/*

RUN apt-get install -y libjemalloc-dev

# Copy the lddl source code to /workspace/lddl in the image, then install.
WORKDIR /workspace/lddl
ADD . .
RUN pip install ./
RUN pip install h5py
RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Download the NLTK model data.
RUN python -m nltk.downloader punkt
