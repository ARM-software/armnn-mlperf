# [MLPerf Inference - Object Detection - ArmNN-TFLite (Debian 9)](https://hub.docker.com/r/ctuning/object-detection-armnn-tflite.debian-9)

1. [Default image](#image_default) (based on [Debian](https://hub.docker.com/_/debian/) 9 latest)
    - [Download](#image_default_download) or [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Object Detection (default command)](#image_default_run_default)
        - [Object Detection (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_default"></a>
## Default image

<a name="image_default_download"></a>
### Download
```
$ docker pull ctuning/object-detection-armnn-tflite.debian-9
```

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:object-detection-armnn-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:object-detection-armnn-tflite.debian-9`
$ docker build -f Dockerfile -t ctuning/object-detection-armnn-tflite.debian-9 .
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Object Detection (default command)

##### Regular NMS; 50 images
```bash
$ ck run docker:object-detection-armnn-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ docker run --rm ctuning/object-detection-armnn-tflite.debian-9 \
    "ck run program:object-detection-armnn-tflite \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=regular \
        --dep_add_tags.dataset=coco.2017,first.50 --env.CK_BATCH_COUNT=50 \
    "
...
Summary:
-------------------------------
All images loaded in 0.138678s
Average image load time: 0.002774s
All images detected in 1283.481567s
Average detection time: 25.633888s
mAP: 0.2973506730812673
Recall: 0.3064842506013031
--------------------------------
```

<a name="image_default_run_custom"></a>
#### Object Detection (custom command)

##### Fast NMS; 50 images
```bash
$ docker run --rm ctuning/object-detection-armnn-tflite.debian-9 \
    "ck run program:object-detection-armnn-tflite \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=fast \
        --dep_add_tags.dataset=coco.2017,first.50 --env.CK_BATCH_COUNT=50 \
    "
...
Summary:
-------------------------------
All images loaded in 0.136716s
Average image load time: 0.002734s
All images detected in 1231.462524s
Average detection time: 24.573481s
mAP: 0.29687329696309245
Recall: 0.30644766969653536
--------------------------------
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm ctuning/object-detection-armnn-tflite.debian-9 bash
```
