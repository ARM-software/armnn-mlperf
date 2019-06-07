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

# Download the MobileNet TF/TFLite models (non-quantized and quantized).
# https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md#install-the-mobilenet-models-for-tflite
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,non-quantized,from-zenodo
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,quantized,from-google
# Download the ResNet TFLite models (with and without the ArgMax operator).
# https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md#install-the-resnet-model
RUN ck install package --tags=image-classification,model,tflite,mlperf,resnet,downloaded,with-argmax
RUN ck install package --tags=image-classification,model,tflite,mlperf,resnet,downloaded,no-argmax

# Install C/C++ dependencies of ArmNN.
RUN ck pull repo --url=https://github.com/arm-software/armnn-mlperf
RUN ck install package --tags=lib,protobuf-host,v3.5
RUN ck install package --tags=lib,flatbuffers
RUN ck install package --tags=lib,boost,for-armnn,static,v1.64\
 && ck virtual env --tags=lib,boost,for-armnn,static,v1.64 --shell_cmd='rm -rf $CK_ENV_LIB_BOOST/../boost_1_64_0'
# Reduce the image size by detecting the TF source in the installed TFLite package rather than installing the TF source package with '--tags=tensorflow,source'.
RUN echo "1.13.1" | ck detect soft:lib.tensorflow.source --full_path=`ck locate env --tags=lib,tflite,v1.13.1`/src/tensorflow/tensorflow.bzl
RUN ck install package --tags=lib,armnn,tflite,rel.19.05

# Download and preprocess the first 500 images of the ImageNet 2012 validation dataset.
RUN ck install package --tags=dataset,imagenet,aux
RUN ck install package --tags=dataset,imagenet,val,original,min --no_tags=resized
RUN ck install package --tags=dataset,imagenet,val,preprocessed

# Compile the Image Classification ArmNN-TFLite program.
RUN ck compile armnn-mlperf:program:image-classification-armnn-tflite

# Compile the Image Classification TFLite program.
RUN ck compile ck-tensorflow:program:image-classification-tflite

# Run the Image Classification ArmNN-TFLite program
# with the non-quantized MobileNet model once.
CMD ["ck run armnn-mlperf:program:image-classification-armnn-tflite \
--dep_add_tags.weights=mobilenet,non-quantized"]
