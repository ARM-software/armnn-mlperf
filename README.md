# [Arm NN](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn) ports for [MLPerf Inference](https://github.com/mlperf/inference) benchmarks

- [Getting started](#getting_started)
    - [Pull CK repositories](#repos)
    - [Install TFLite](#tflite)
    - [Install ArmNN](#armnn)
        - [with TFLite and Neon support](#armnn_tflite_neon)
        - [with TFLite and OpenCL support](#armnn_tflite_opencl)
        - [with TFLite and Reference support](#armnn_tflite_reference)
    - [Download ImageNet validation dataset](#imagenet)
- [MobileNet model](#mobilenet)
    - [TFLite data](#mobilenet_tflite) (reference)
    - [ArmNN Neon data](#mobilenet_armnn_neon)
    - [ArmNN OpenCL data](#mobilenet_armnn_opencl)
    - [ArmNN Reference data](#mobilenet_armnn_reference)
- [ResNet model](#resnet)
    - [TFLite data](#resnet_tflite) (reference)
    - [ArmNN Neon data](#resnet_armnn_neon)
    - [ArmNN OpenCL data](#resnet_armnn_opencl)
    - [ArmNN Reference data](#resnet_armnn_reference)

<a name="getting_started"></a>
# Getting started

<a name="repos"></a>
## Pull CK repositories
```
$ ck pull repo:armnn-mlperf --url=git@github.com:ARM-software/mlperf.git
$ ck pull repo:ck-mlperf
$ ck pull repo:ck-armnn
```

<a name="tflite"></a>
## Install TFLite
```
$ ck install package --tags=lib,tflite,v1.13
```
**NB:** For more details, please refer to the
[TFLite installation instructions](https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md#install-tensorflow-lite-tflite).

<a name="armnn"></a>
## Install ArmNN

To install ArmNN with full support (frontends: TF, TFLite, ONNX; backends: Reference, OpenCL, Neon):
```
$ ck install package --tags=lib,armnn,tf,tflite,onnx,neon,opencl,rel.19.02
```
**NB:** For more details, please refer to the [CK-ArmNN](http://github.com/ctuning/ck-armnn) repository.

If you would like to save time, you can only build with TFLite support as below.
<a name="armnn_tflite_reference"></a>
### Install ArmNN with TFLite and Reference support
```
$ ck install package --tags=lib,armnn,tflite,rel.19.02
```
<a name="armnn_tflite_neon"></a>
### Install ArmNN with TFLite and Neon support
```
$ ck install package --tags=lib,armnn,tflite,neon,rel.19.02
```
<a name="armnn_tflite_opencl"></a>
### Install ArmNN with TFLite and OpenCL support
```
$ ck install package --tags=lib,armnn,tflite,opencl,rel.19.02
```

<a name="imagenet"></a>
## Download ImageNet validation dataset

### Full (50,000 images)
```
$ ck install package:imagenet-2012-val
```

### Minimal (500 images)
```
$ ck install package:imagenet-2012-val-min
```

<a name="mobilenet"></a>
## MobileNet

Install the MobileNet model:
```
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized
```

<a name="mobilenet_tflite"></a>
### TFLite data (reference)

#### Run on 500 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-accuracy-500 \
--tags=image-classification,mlperf,mobilenet,tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** You can also run the same command on the full ImageNet validation dataset of 50,000 images (see below).

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-accuracy \
--tags=image-classification,mlperf,mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** On the first run, the dataset will be preprocessed and cached in the
`preprocessed/` subdirectory under the program's directory:
```
$ du -hs -L `ck find program:image-classification-tflite`/preprocessed
7.1G
```

You may want to create a symbolic link to this directory for the corresponding ArmNN program e.g.:
```
$ ln -s `ck find program:image-classification-tflite`/preprocessed \
        `ck find program:image-classification-armnn-tflite`/preprocessed
```

<a name="mobilenet_armnn_neon"></a>
### ArmNN Neon data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-500-neon \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,500,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-neon \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_armnn_opencl"></a>
### ArmNN OpenCL data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-500-opencl \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,500,opencl \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-opencl \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,opencl \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_armnn_reference"></a>
### ArmNN Reference data (**NOT RECOMMENDED**)

**NB:** This validation can run on x86 or arm. However, it is completely unoptimised and hence extremely slow (e.g. 6.5 seconds per image on HiKey960 or 2.9 seconds per image on a Xeon server).

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-500 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet"></a>
## ResNet

Install the ResNet model:
```
$ ck install package --tags=model,tflite,mlperf,resnet
```

<a name="resnet_tflite"></a>
### TFLite data (reference)

#### Run on 500 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-tflite-accuracy-500 \
--tags=image-classification,mlperf,resnet,tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** You can also run the same command on the full ImageNet validation dataset of 50,000 images (see below).

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-tflite-accuracy \
--tags=image-classification,mlperf,resnet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_neon"></a>
### ArmNN Neon data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-500-neon \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,500,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-neon \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_opencl"></a>
### ArmNN OpenCL data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-500-opencl \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,500,opencl \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-opencl \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,opencl \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_reference"></a>
### ArmNN Reference data (**NOT RECOMMENDED**)

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-500 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
