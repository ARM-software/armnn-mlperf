# ArmNN-TFLite image classification program

## Compile with a particular backend

### Reference
```
$ ck compile program:image-classification-armnn-tflite
```

### Neon
```
$ ck compile program:image-classification-armnn-tflite --env.USE_NEON
```

### OpenCL
```
$ ck compile program:image-classification-armnn-tflite --env.USE_OPENCL
```

### Neon and OpenCL
```
$ ck compile program:image-classification-armnn-tflite --env.USE_NEON --env.USE_OPENCL
```

## Run

**NB:** Must use the same backend options as for compilation.

### Neon
```
$ ck run program:image-classification-armnn-tflite \
--env.CK_BATCH_COUNT=5 \
--env.USE_NEON
```
**where:**
 - `CK_BATCH_COUNT` - the number of batches to evaluate (1 by default).
 - `USE_NEON` - enable CPU acceleration (false by default).
 - `USE_OPENCL` - enable GPU acceleration (false by default).

## Benchmark

**NB:** [Similar instructions](https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md#benchmarking-instructions) are used to benchmark [`program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite).

### Benchmark the performance

#### Neon

```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON=1 \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=1 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-v1-1.00-224-armnn-tflite-performance-neon \
--tags=mlperf,image-classification,mobilenet-v1-1.0-224,armnn-tflite,performance,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

### Benchmark the accuracy

#### Neon

```
$ ck benchmark program:image-classification-armnn-tflite --env.USE_NEON=1 \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-v1-1.00-224-armnn-tflite-accuracy-neon \
--tags=mlperf,image-classification,mobilenet-v1-1.0-224,armnn-tflite,accuracy,neon \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

