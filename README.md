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
    - [Validate data](#mobilenet_validate)
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
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-accuracy-50000 \
--tags=image-classification,mlperf,mobilenet,tflite,accuracy,50000 \
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
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-neon-500 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,neon,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-50000-neon \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,neon,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_armnn_opencl"></a>
### ArmNN OpenCL data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-opencl-500 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,opencl,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-opencl-50000 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,opencl,50000 \
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
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-50000 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_validate"></a>
### Validate experimental data

To validate the equivalence of the optimized ArmNN implementation versus the reference TFLite one,
we collected experimental data as above on two platforms:
- A Linaro [HiKey960](https://www.96boards.org/product/hikey960/) board (`hikey`): TFLite vs. ArmNN Neon vs. ArmNN OpenCL.
- A Intel Xeon server (`velociti`): TFLite vs. ArmNN Reference.

The resulting experimental entries were archived e.g. as follows:
```bash
hikey$ ck list local:experiment:mlperf-mobilenet*accuracy*500
mlperf-mobilenet-tflite-accuracy-500
mlperf-mobilenet-armnn-tflite-accuracy-neon-500
mlperf-mobilenet-armnn-tflite-accuracy-opencl-500
hikey$ ck zip local:experiment:mlperf-mobilenet*accuracy*500 \
                --archive_name=mlperf-mobilenet-accuracy-500-hikey.zip
```
The archives were then uploaded to DropBox.
You can follow instructions below to download the archives and validate the accuracy.

#### `hikey`
##### 500 images
```bash
$ wget https://www.dropbox.com/s/9lz7yncy1xtqlvj/mlperf-mobilenet-accuracy-500-hikey.zip
$ ck add repo --zip=mlperf-mobilenet-accuracy-500-hikey.zip
$ ck list --repo_uoa=mlperf-mobilenet-accuracy-500-hikey --print_full
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-neon-500
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-tflite-accuracy-500
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-opencl-500
```
###### TFLite vs. ArmNN Neon
```bash
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-tflite-accuracy-500 \
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-neon-500
...
{'epsilon': 1e-05,
 'max_delta': 7.000000000034756e-06,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 0,
 'num_mismatched_probabilities': 0,
 'return': 0}
```
###### TFLite vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-tflite-accuracy-500 \
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-opencl-500
...
{'epsilon': 1e-05,
 'max_delta': 9.000000000036756e-06,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 0,
 'num_mismatched_probabilities': 0,
 'return': 0}
```
###### ArmNN Neon vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-neon-500 \
mlperf-mobilenet-accuracy-500-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-opencl-500
...
{'epsilon': 1e-05,
 'max_delta': 6.0000000000060005e-06,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 0,
 'num_mismatched_probabilities': 0,
 'return': 0}
```

##### 50,000 images
```bash
$ wget https://www.dropbox.com/s/3cdi3lx7jfxwse7/mlperf-mobilenet-accuracy-50000-hikey.zip
$ ck add repo --zip=mlperf-mobilenet-accuracy-50000-hikey.zip
$ ck list --repo_uoa=mlperf-mobilenet-accuracy-50000-hikey --print_full
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-neon-50000
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-tflite-accuracy-50000
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-opencl-50000
```
###### TFLite vs. ArmNN Neon
```bash
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-tflite-accuracy-50000 \
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-neon-50000
...
{'epsilon': 1e-05,
 'max_delta': 1.3000000000040757e-05,
 'num_mismatched_classes': 10,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 20,
 'num_mismatched_probabilities': 19,
 'return': 0}
```
###### TFLite vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-tflite-accuracy-50000 \
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-opencl-50000
...
{'epsilon': 1e-05,
 'max_delta': 1.4000000000014001e-05,
 'num_mismatched_classes': 8,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 20,
 'num_mismatched_probabilities': 18,
 'return': 0}
```
###### ArmNN Neon vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-neon-50000 \
mlperf-mobilenet-accuracy-50000-hikey:experiment:mlperf-mobilenet-armnn-tflite-accuracy-opencl-50000
...
Checking ILSVRC2012_val_00033823.JPEG...
- mismatched classes at index 2: 137 != 136
- mismatched classes at index 3: 136 != 137
...
{'epsilon': 1e-05,
 'max_delta': 8.000000000008e-06,
 'num_mismatched_classes': 2,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 1,
 'num_mismatched_probabilities': 0,
 'return': 0}
```

#### `velociti`
##### 500 images
```bash
$ wget https://www.dropbox.com/s/j2rdh3uzhz3lqh7/mlperf-mobilenet-accuracy-500-velociti.zip
$ ck add repo --zip=mlperf-mobilenet-accuracy-500-velociti.zip
$ ck list --repo_uoa=mlperf-mobilenet-accuracy-500-velociti --print_full
mlperf-mobilenet-accuracy-500-velociti:experiment:mlperf-mobilenet-armnn-tflite-accuracy-500
mlperf-mobilenet-accuracy-500-velociti:experiment:mlperf-mobilenet-tflite-accuracy-500
```
###### TFLite vs. ArmNN Reference
```
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-500-velociti:experiment:mlperf-mobilenet-armnn-tflite-accuracy-500 \
mlperf-mobilenet-accuracy-500-velociti:experiment:mlperf-mobilenet-tflite-accuracy-500
...
{'epsilon': 1e-05,
 'max_delta': 7.000000000090267e-06,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 0,
 'num_mismatched_probabilities': 0,
 'return': 0}
```

##### 50,000 images
```bash
$ wget https://www.dropbox.com/s/z5bx7aeocwdyrww/mlperf-mobilenet-accuracy-50000-velociti.zip
$ ck add repo --zip=mlperf-mobilenet-accuracy-50000-velociti.zip
$ ck list --repo_uoa=mlperf-mobilenet-accuracy-50000-velociti --print_full
mlperf-mobilenet-accuracy-50000-velociti:experiment:mlperf-mobilenet-armnn-tflite-accuracy-50000
mlperf-mobilenet-accuracy-50000-velociti:experiment:mlperf-mobilenet-tflite-accuracy-50000
```
###### TFLite vs. ArmNN Reference
```
$ ck compare_experiments mlperf \
mlperf-mobilenet-accuracy-50000-velociti:experiment:mlperf-mobilenet-armnn-tflite-accuracy-50000 \
mlperf-mobilenet-accuracy-50000-velociti:experiment:mlperf-mobilenet-tflite-accuracy-50000
...
{'epsilon': 1e-05,
 'max_delta': 1.2000000000012001e-05,
 'num_mismatched_classes': 2,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 14,
 'num_mismatched_probabilities': 17,
 'return': 0}
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
--record --record_repo=local --record_uoa=mlperf-resnet-tflite-accuracy-50000 \
--tags=image-classification,mlperf,resnet,tflite,accuracy,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_neon"></a>
### ArmNN Neon data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-neon-500 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,neon,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-50000-neon \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,neon,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_opencl"></a>
### ArmNN OpenCL data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-opencl-500 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,opencl,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-opencl-50000 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,opencl,50000 \
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
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-50000 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
