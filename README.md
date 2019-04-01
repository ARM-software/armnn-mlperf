## Getting started

To pull this CK repository, run:
```
$ ck pull repo:armnn-mlperf --url=git@github.com:ARM-software/mlperf.git
```

1. [CK repositories](#repos)
1. [ImageNet validation dataset](#imagenet)
1. [MobileNet model](#mobilenet)
    i. [TFLite data](#mobilenet_tflite) (reference)
    i. [ArmNN Neon data](#mobilenet_armnn_neon)
    i. [ArmNN OpenCL data](#mobilenet_armnn_opencl)
    i. [ArmNN Reference data](#mobilenet_armnn_reference)


<a name="repos"></a>
## Install CK repositories
```
$ ck pull repo:ck-mlperf
$ ck pull repo:ck-armnn
```

<a name="imagenet"></a>
## Install ImageNet validation dataset

### Full (50,000 images)
```
$ ck install package:imagenet-2012-val
```

### Minimal (500 images)
```
$ ck install package:imagenet-2012-val-min
```

<a name="mobilenet"></a>
## Install MobileNet model
```
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized
```

<a name="mobilenet_tflite"></a>
### TFLite data (reference)

For full details, see the [MLPerf Inference](https://github.com/mlperf/inference) instructions
for the [MobileNet TFLite reference implementation](https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md).
A brief summary is provided below.

#### Install TFLite
```
$ ck install package --tags=lib,tflite,v1.13
```

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

For full details on how to build ArmNN with Neon support, please refer to the [CK-ArmNN](http://github.com/ctuning/ck-armnn) repository.
A brief summary is provided below.

#### Install ArmNN with Neon and TFLite support
Minimally, run:
```
$ ck install package --tags=lib,armnn,tflite,neon,rel.19.02
```
or also include OpenCL, TF, ONNX support e.g.:
```
$ ck install package --tags=lib,armnn,tflite,neon,opencl,tf,onnx,rel.19.02
```

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

For full details on how to build ArmNN with OpenCL support, please refer to the [CK-ArmNN](http://github.com/ctuning/ck-armnn) repository.
A brief summary is provided below.

#### Install ArmNN with OpenCL and TFLite support
Minimally, run:
```
$ ck install package --tags=lib,armnn,tflite,opencl,rel.19.02
```
or also include Neon, TF, ONNX support e.g.:
```
$ ck install package --tags=lib,armnn,tflite,neon,opencl,tf,onnx,rel.19.02
```

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
