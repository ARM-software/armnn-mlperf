# [Arm NN](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn) ports for [MLPerf Inference](https://github.com/mlperf/inference) benchmarks

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)

- [Getting started](#gs)
    - [Install CK](#gs_ck)
    - [Pull CK repositories](#gs_repos)
    - [Install TFLite](#gs_tflite)
    - [Install ArmNN](#gs_armnn)
        - [with TFLite, Neon and OpenCL support](#gs_armnn_tflite_neon_opencl) (**recommended**)
        - [with TFLite and Neon support](#gs_armnn_tflite_neon)
        - [with TFLite and OpenCL support](#gs_armnn_tflite_opencl)
        - [with TFLite and Reference support](#gs_armnn_tflite_reference)
- [Image classification](#image_classification)
    - [Download the ImageNet 2012 validation dataset](#imagenet)
    - [MobileNet](#mobilenet)
        - [Model](#mobilenet_model)
        - [TFLite data](#mobilenet_tflite) (reference)
        - [ArmNN Neon data](#mobilenet_armnn_neon)
        - [ArmNN OpenCL data](#mobilenet_armnn_opencl)
        - [ArmNN Reference data](#mobilenet_armnn_reference)
        - [Validate data](#mobilenet_validate)
    - [ResNet](#resnet)
        - [Model](#resnet_model)
        - [TFLite data](#resnet_tflite) (reference)
        - [ArmNN Neon data](#resnet_armnn_neon)
        - [ArmNN OpenCL data](#resnet_armnn_opencl)
        - [ArmNN Reference data](#resnet_armnn_reference)
        - [Validate data](#resnet_validate)
- [Object detection](#object_detection)
    - [Caveats](#object_detection_caveats)
    - [Download the COCO 2017 validation dataset](#coco)
    - [SSD-MobileNet](#ssd_mobilenet)
        - [Model](#ssd_mobilenet_model)
        - [TFLite data](#ssd_mobilenet_tflite) (reference)
        - [ArmNN Neon data](#ssd_mobilenet_armnn_neon)
        - [ArmNN OpenCL data](#ssd_mobilenet_armnn_opencl)
        - [Validate data](#ssd_mobilenet_validate)


<a name="gs"></a>
# Getting started

<a name="gs_ck"></a>
## Install CK
Please follow the [CK installation instructions](https://github.com/ctuning/ck#installation).

<a name="gs_repos"></a>
## Pull CK repositories
```
$ ck pull repo --url=https://github.com/arm-software/armnn-mlperf
$ ck list repo:*armnn*
ck-armnn
armnn-mlperf
$ ck list repo:*mlperf*
ck-mlperf
armnn-mlperf
```
**NB:** Remember to refresh all the repositories after any updates (e.g. bug fixes):
```
$ ck pull all
```

<a name="gs_tflite"></a>
## Install TFLite
```
$ ck install package --tags=lib,tflite,v1.13
```

<a name="gs_armnn"></a>
## Install ArmNN
To install ArmNN with full support (frontends: TF, TFLite, ONNX; backends: Reference, OpenCL, Neon):
```
$ ck install package --tags=lib,armnn,tf,tflite,onnx,neon,opencl,rel.19.02
```
**NB:** On a platform with only a couple of GB of RAM, you way wish to restrict
the number of CPU build threads e.g. as follows:
```
$ ck install package --tags=lib,armnn,tflite,neon,opencl,rel.19.02 \
--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=4
```

If you would like to save time, you can build with TFLite frontend support only using one of the backend options below.
For more details, please refer to the [CK-ArmNN](http://github.com/ctuning/ck-armnn) repository.

<a name="gs_armnn_tflite_neon_opencl"></a>
### Option 1: Install ArmNN with TFLite, Neon and OpenCL support (**recommended**)
```
$ ck install package --tags=lib,armnn,tflite,neon,opencl,rel.19.02
```
<a name="gs_armnn_tflite_neon"></a>
### Option 2: Install ArmNN with TFLite and Neon support
```
$ ck install package --tags=lib,armnn,tflite,neon,rel.19.02
```
<a name="gs_armnn_tflite_opencl"></a>
### Option 3: Install ArmNN with TFLite and OpenCL support
```
$ ck install package --tags=lib,armnn,tflite,opencl,rel.19.02
```
<a name="gs_armnn_tflite_reference"></a>
### Option 4: ArmNN with TFLite and Reference support
```
$ ck install package --tags=lib,armnn,tflite,rel.19.02
```

<a name="image_classification"></a>
# Image classification

Please follow the MLPerf image classification instructions to install dependencies such as Python packages: first, the [common instructions](https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/README.md); then, the [TFLite instructions](https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md).

<a name="imagenet"></a>
## Download the ImageNet 2012 validation dataset

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

<a name="mobilenet_model"></a>
### Model
Install the MobileNet model:
```
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized
```

<a name="mobilenet_tflite"></a>
### TFLite data (reference)

#### Run on 500 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-accuracy-500 \
--tags=image-classification,mlperf,mobilenet,tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** You can also run the same command on the full ImageNet validation dataset of 50,000 images (see below).

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
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
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-neon-500 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,neon,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-neon-50000 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,neon,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_armnn_opencl"></a>
### ArmNN OpenCL data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-opencl-500 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,opencl,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-opencl-50000 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,opencl,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_armnn_reference"></a>
### ArmNN Reference data (**NOT RECOMMENDED**)

**NB:** This validation can run on x86 or arm. However, it is completely unoptimized and hence extremely slow (e.g. 6.5 seconds per image on a Linaro HiKey960 board or 2.9 seconds per image on an Intel Xeon server).

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-500 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-armnn-tflite-accuracy-50000 \
--tags=image-classification,mlperf,mobilenet,armnn-tflite,accuracy,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="mobilenet_validate"></a>
### Validate experimental data

To validate the equivalence of the optimized ArmNN implementation versus the reference TFLite one,
we collected experimental data as above on two platforms:
- A Linaro [HiKey960](https://www.96boards.org/product/hikey960/) board (`hikey`): TFLite vs. ArmNN Neon vs. ArmNN OpenCL (500 and 50,000 images).
- An Intel Xeon server (`velociti`): TFLite vs. ArmNN Reference (500 and 50,000 images).

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

<a name="resnet_model"></a>
### Model
Install the ResNet model:
```
$ ck install package --tags=model,tflite,mlperf,resnet

More than one package or version found:

 0) model-tflite-mlperf-resnet-no-argmax  Version 1.5  (afb43014ef38f646)
 1) model-tflite-mlperf-resnet  Version 1.5  (d60d4e9a84151271)
 2) model-tflite-convert-from-tf (35e84375ac48dcb1), Variations: resnet

Please select the package to install [ hit return for "0" ]:
```

Option 0 will download a TFLite model preconverted from the TF model.  During
the conversion, the `ArgMax` operator causing an
[issue](https://github.com/ARM-software/armnn/issues/150) with ArmNN v19.02 was
excluded.

Option 1 will download a TFLite model preconverted from the TF model, but
including the `ArgMax` operator. This variant can be used once the above issue
is resolved.

Option 2 will download the TF model and convert it to TFLite, while excluding
the `ArgMax` operator.  Since the conversion relies on a prebuilt version of
TF, this option is only viable on x86.  (This constraint can be relaxed, but
building TF on Arm [is not officially
supported](https://github.com/tensorflow/tensorflow/issues/25607#issuecomment-466583730).)

<a name="resnet_tflite"></a>
### TFLite data (reference)

#### Run on 500 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-tflite-accuracy-500 \
--tags=image-classification,mlperf,resnet,tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** You can also run the same command on the full ImageNet validation dataset of 50,000 images (see below).

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-tflite-accuracy-50000 \
--tags=image-classification,mlperf,resnet,tflite,accuracy,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_neon"></a>
### ArmNN Neon data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-neon-500 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,neon,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-neon-50000 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,neon,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_opencl"></a>
### ArmNN OpenCL data

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-opencl-500 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,opencl,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images
```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_OPENCL \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-opencl-50000 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,opencl,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_armnn_reference"></a>
### ArmNN Reference data (**NOT RECOMMENDED**)

**NB:** This validation can run on x86 or arm. However, it is completely unoptimized and hence extremely slow (e.g. 34.8 seconds per image on a Linaro HiKey960 board or 16.9 seconds per image on an Intel Xeon server).

#### Run on 500 images
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-500 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,500 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

#### Run on 50,000 images (**NOT RUN**)
```
$ ck benchmark program:image-classification-armnn-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-resnet-armnn-tflite-accuracy-50000 \
--tags=image-classification,mlperf,resnet,armnn-tflite,accuracy,50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="resnet_validate"></a>
### Validate experimental data

To validate the equivalence of the optimized ArmNN implementation versus the reference TFLite one,
we collected experimental data as above on two platforms:
- A Linaro [HiKey960](https://www.96boards.org/product/hikey960/) board (`hikey`): TFLite vs. ArmNN Neon vs. ArmNN OpenCL (500 and 50,000 images).
- An Intel Xeon server (`velociti`): TFLite vs. ArmNN Reference (500 images only).

The resulting experimental entries were archived e.g. as follows:
```bash
hikey$ ck list local:experiment:mlperf-resnet*accuracy*500
mlperf-resnet-tflite-accuracy-500
mlperf-resnet-armnn-tflite-accuracy-neon-500
mlperf-resnet-armnn-tflite-accuracy-opencl-500
hikey$ ck zip local:experiment:mlperf-resnet*accuracy*500 \
                --archive_name=mlperf-resnet-accuracy-500-hikey.zip
```
The archives were then uploaded to DropBox.
You can follow instructions below to download the archives and validate the accuracy.

#### `hikey`
##### 500 images
```bash
$ wget https://www.dropbox.com/s/eod0bflxxzpudmr/mlperf-resnet-accuracy-500-hikey.zip
$ ck add repo --zip=mlperf-resnet-accuracy-500-hikey.zip
$ ck list --repo_uoa=mlperf-resnet-accuracy-500-hikey --print_full
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-neon-500
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-opencl-500
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-tflite-accuracy-500
```
###### TFLite vs. ArmNN Neon
```bash
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-tflite-accuracy-500 \
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-neon-500
...
{'epsilon': 1e-05,
 'max_delta': 1.0999999999983245e-05,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 2,
 'num_mismatched_probabilities': 3,
 'return': 0}
```
###### TFLite vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-tflite-accuracy-500 \
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-opencl-500
...
{'epsilon': 1e-05,
 'max_delta': 1.0999999999983245e-05,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 3,
 'num_mismatched_probabilities': 4,
 'return': 0}
```
###### ArmNN Neon vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-neon-500 \
mlperf-resnet-accuracy-500-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-opencl-500
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
$ wget https://www.dropbox.com/s/1yzv6unriqs18yb/mlperf-resnet-accuracy-50000-hikey.zip
$ ck add repo --zip=mlperf-resnet-accuracy-50000-hikey.zip
$ ck list --repo_uoa=mlperf-resnet-accuracy-50000-hikey --print_full
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-opencl-50000
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-neon-50000
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-tflite-accuracy-50000
```
###### TFLite vs. ArmNN Neon
```bash
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-tflite-accuracy-50000 \
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-neon-50000
...
{'epsilon': 1e-05,
 'max_delta': 2.4000000000024002e-05,
 'num_mismatched_classes': 6,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 107,
 'num_mismatched_probabilities': 153,
 'return': 0}
```
###### TFLite vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-tflite-accuracy-50000 \
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-opencl-50000
...
{'epsilon': 1e-05,
 'max_delta': 2.5000000000052758e-05,
 'num_mismatched_classes': 4,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 147,
 'num_mismatched_probabilities': 190,
 'return': 0}
```
###### ArmNN Neon vs. ArmNN OpenCL
```bash
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-neon-50000 \
mlperf-resnet-accuracy-50000-hikey:experiment:mlperf-resnet-armnn-tflite-accuracy-opencl-50000
...
{'epsilon': 1e-05,
 'max_delta': 1.0000000000010001e-05,
 'num_mismatched_classes': 6,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 5,
 'num_mismatched_probabilities': 2,
 'return': 0}
```

#### `velociti`
##### 500 images
```bash
$ wget https://www.dropbox.com/s/1jv4lpfp1ddr2j7/mlperf-resnet-accuracy-500-velociti.zip
$ ck add repo --zip=mlperf-resnet-accuracy-500-velociti.zip
$ ck list --repo_uoa=mlperf-resnet-accuracy-500-velociti --print_full
mlperf-resnet-accuracy-500-velociti:experiment:mlperf-resnet-armnn-tflite-accuracy-500
mlperf-resnet-accuracy-500-velociti:experiment:mlperf-resnet-tflite-accuracy-500
```
###### TFLite vs. ArmNN Reference
```
$ ck compare_experiments mlperf \
mlperf-resnet-accuracy-500-velociti:experiment:mlperf-resnet-armnn-tflite-accuracy-500 \
mlperf-resnet-accuracy-500-velociti:experiment:mlperf-resnet-tflite-accuracy-500
...
{'epsilon': 1e-05,
 'max_delta': 3.0000000000030003e-06,
 'num_mismatched_classes': 0,
 'num_mismatched_elementary_keys': 0,
 'num_mismatched_files': 0,
 'num_mismatched_probabilities': 0,
 'return': 0}
```

<a name="object_detection"></a>
# Object detection

Please follow the MLPerf object detection instructions to install dependencies such as Python packages: first, the [common instructions](https://github.com/mlperf/inference/blob/master/edge/object_detection/ssd_mobilenet/README.md); then, the [TFLite instructions](https://github.com/mlperf/inference/blob/master/edge/object_detection/ssd_mobilenet/tflite/README.md).

<a name="object_detection_caveats"></a>
## Caveats

### TFLite
The SSD models require TFLite 1.13.1.

### Python 3

The [COCO API](https://github.com/cocodataset/cocoapi) (used to evaluate object detection accuracy on the [COCO dataset](http://cocodataset.org/)) requires Python 3. Since many embedded platforms use Python 2 by default (including HiKey960), [care must be taken](https://github.com/dividiti/inference/blob/ssd_mobilenet/edge/object_detection/ssd_mobilenet/README.md#install-python-3-and-the-latest-pip) not to mix Python 3 and Python 2 packages. Therefore, all benchmarking commands below use the `CK_PYTHON=python3` prefix to ensure CK runs under Python 3.

<a name="coco"></a>
## Download and preprocess the COCO 2017 validation dataset
```bash
$ ck install package --tags=object-detection,dataset,coco.2017,val,original,full
$ ck install package --tags=object-detection,dataset,coco.2017,preprocessed,full
```

<a name="ssd_mobilenet"></a>
## SSD-MobileNet

<a name="ssd_mobilenet_model"></a>
### Model
Install the SSD-MobileNet model:
```bash
$ ck install package --tags=model,tflite,mlperf,object-detection,ssd-mobilenet
```

<a name="ssd_mobilenet_tflite"></a>
### TFLite data (reference)
```bash
$ CK_PYTHON=python3 ck benchmark program:object-detection-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="ssd_mobilenet_armnn_neon"></a>
### ArmNN Neon data
```bash
$ CK_PYTHON=python3 ck benchmark program:object-detection-armnn-tflite --env.USE_NMS=regular --env.USE_NEON \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-neon \
--tags=mlperf,object-detection,ssd-mobilenet,armnn-tflite,accuracy,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="ssd_mobilenet_armnn_opencl"></a>
### ArmNN OpenCL data
```bash
$ CK_PYTHON=python3 ck benchmark program:object-detection-armnn-tflite --env.USE_NMS=regular --env.USE_OPENCL \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-opencl \
--tags=mlperf,object-detection,ssd-mobilenet,armnn-tflite,accuracy,opencl \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="ssd_mobilenet_validate"></a>
### Validate experimental data

To validate the equivalence of the optimized ArmNN implementation versus the reference TFLite one,
we collected experimental data as above on:
- A Linaro [HiKey960](https://www.96boards.org/product/hikey960/) board (`hikey`): TFLite vs. ArmNN Neon vs. ArmNN OpenCL.

The resulting experimental entries were archived e.g. as follows:
```bash
hikey$ ck list local:experiment:mlperf-object-detection-ssd-mobilenet*accuracy*
...
hikey$ ck zip local:experiment:mlperf-object-detection-ssd-mobilenet*accuracy* \
                --archive_name=mlperf-object-detection-ssd-mobilenet-accuracy-hikey.zip
```
The archives were then uploaded to DropBox.
You can follow instructions below to download the archives and validate the accuracy.

#### hikey
```
$ wget https://www.dropbox.com/s/jzpum9fedwgq8rd/mlperf-object-detection-ssd-mobilenet-accuracy-hikey.zip
$ ck add repo --zip=mlperf-object-detection-ssd-mobilenet-accuracy-hikey.zip
$ ck list --repo_uoa=mlperf-object-detection-ssd-mobilenet-accuracy-hikey --print_full
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-opencl
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-neon
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-tflite-accuracy
```

##### TFLite vs. ArmNN Neon
```
$ ck compare_experiments mlperf \
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-neon
```

##### TFLite vs. ArmNN OpenCL
```
$ ck compare_experiments mlperf \
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-opencl
```

##### ArmNN Neon vs. ArmNN OpenCL
```
$ ck compare_experiments mlperf \
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-neon \
mlperf-object-detection-ssd-mobilenet-accuracy-hikey:experiment:mlperf-object-detection-ssd-mobilenet-armnn-tflite-accuracy-opencl
```
