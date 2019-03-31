## Getting started

To pull this CK repository, run:
```
$ ck pull repo:armnn-mlperf --url=git@github.com:ARM-software/mlperf.git
```

## MobileNet

### TFLite reference data

For full details, see the [MLPerf Inference](https://github.com/mlperf/inference) instructions
for the [MobileNet TFLite reference implementation](https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md).
A brief summary is provided below.

#### Install TFLite and MobileNet
```
$ ck pull repo:ck-mlperf
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized
$ ck install package --tags=lib,tflite,v1.13
```

#### Run on 500 images
```
$ ck install package:imagenet-2012-val-min
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-accuracy-500 \
--tags=image-classification,mlperf,mobilenet,tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** You can also run the same command on the full ImageNet validation dataset of 50,000 images (see below).

#### Run on 50,000 images
```
$ ck install package:imagenet-2012-val
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

### ArmNN reference data (**NOT RECOMMENDED**)

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
