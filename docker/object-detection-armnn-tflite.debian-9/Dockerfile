FROM debian:9

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# ArmNN needs: cmake.
RUN apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 git wget zip libz-dev\
 curl\
 cmake\
 python3 python3-pip\
 vim\
 && apt clean

# Create non-root user.
RUN useradd --create-home --user-group --shell /bin/bash dvdt
USER dvdt:dvdt
WORKDIR /home/dvdt

# Install Collective Knowledge (CK).
ENV CK_ROOT=/home/dvdt/CK \
    CK_REPOS=/home/dvdt/CK_REPOS \
    CK_TOOLS=/home/dvdt/CK_TOOLS \
    PATH=${CK_ROOT}/bin:/home/dvdt/.local/bin:${PATH} \
    CK_PYTHON=python3 \
    CK_CC=gcc \
    GIT_USER="dividiti" \
    GIT_EMAIL="info@dividiti.com" \
    LANG=C.UTF-8
RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}
RUN git config --global user.name ${GIT_USER} && git config --global user.email ${GIT_EMAIL}
RUN git clone https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT} \
 && ${CK_PYTHON} setup.py install --user \
 && ${CK_PYTHON} -c "import ck.kernel as ck; print ('Collective Knowledge v%s' % ck.__version__)"

# Pull CK repositories (including ck-env, ck-autotuning and ck-tensorflow).
RUN ck pull repo:ck-mlperf

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

# Detect C/C++ compiler (gcc).
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
# Install TFLite.
RUN ck install package --tags=lib,tensorflow-lite,tensorflow-static,v1.13.1

# Detect Python.
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`
# Install the latest Python package installer (pip).
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user
# Install Python dependencies.
RUN ck install package --tags=lib,python-package,numpy
RUN ck install package --tags=lib,python-package,scipy --force_version=1.2.1
RUN ck install package --tags=lib,python-package,pillow
RUN ck install package --tags=lib,python-package,matplotlib
RUN ck install package --tags=lib,python-package,cython
RUN ck show env --tags=python-package
# Install Python COCO API.
RUN ck install package --tags=tool,coco,api

# Download the SSD-MobileNet TF/TFLite models (non-quantized and quantized).
# https://github.com/mlperf/inference/blob/master/edge/object_detection/ssd_mobilenet/tflite/README.md#install-the-ssd-mobilenet-models-for-tflite
RUN ck install package --tags=model,tflite,object-detection,mlperf,ssd-mobilenet,non-quantized

# Install C/C++ dependencies of ArmNN.
RUN ck pull repo --url=https://github.com/arm-software/armnn-mlperf
RUN ck install package --tags=lib,protobuf-host,v3.5
RUN ck install package --tags=lib,flatbuffers
RUN ck install package --tags=lib,boost,for-armnn,static,v1.64\
 && ck virtual env --tags=lib,boost,for-armnn,static,v1.64 --shell_cmd='rm -rf $CK_ENV_LIB_BOOST/../boost_1_64_0'
# Reduce the image size by detecting the TF source in the installed TFLite package rather than installing the TF source package with '--tags=tensorflow,source'.
RUN echo "1.13.1" | ck detect soft:lib.tensorflow.source --full_path=`ck locate env --tags=lib,tflite,v1.13.1`/src/tensorflow/tensorflow.bzl
RUN ck install package --tags=lib,armnn,tflite,rel.19.05

# Download the COCO 2017 validation dataset (5,000 images) to the default path;
# preprocess only the first 50 images (as CPU Reference is too slow for full evaluation);
# remove the original images and training annotations while taking care to preserve
# 'val2017/000000000139.jpg' used to detect COCO 2017 (ck-env:soft:dataset.coco.2017.val).
RUN echo | ck install package --tags=object-detection,dataset,coco.2017,val,original,full\
 && ck install package --tags=object-detection,dataset,coco.2017,preprocessed,first.50\
 && ck virtual env --tags=object-detection,dataset,coco.2017,val,original,full --shell_cmd=\
'mv $CK_ENV_DATASET_IMAGE_DIR/000000000139.jpg $CK_ENV_DATASET_COCO/;\
 rm $CK_ENV_DATASET_IMAGE_DIR/*;\
 mv $CK_ENV_DATASET_COCO/000000000139.jpg $CK_ENV_DATASET_IMAGE_DIR/;\
 rm $CK_ENV_DATASET_COCO_LABELS_DIR/*train2017.json'

# Compile the Object Detection ArmNN-TFLite program.
RUN ck compile program:object-detection-armnn-tflite

# Compile the Object Detection TFLite program.
RUN ck compile program:object-detection-tflite

# Run the Object Detection TFLite program
# with the non-quantized SSD-MobileNet model with regular non-max-suppression
# on the first 50 images of the COCO 2017 validation dataset.
CMD ["ck run program:object-detection-armnn-tflite \
--dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=regular \
--dep_add_tags.dataset=coco.2017,first.50 --env.CK_BATCH_COUNT=50"]
