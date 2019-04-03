# TensorFlow Lite image classification program

This program uses a statically linked [TensorFlow Lite](https://www.tensorflow.org/lite/) library.

## Compile (depending on desired backend)

```
$ ck compile program:image-classification-armnn-tflite
```

```
$ ck compile program:image-classification-armnn-tflite --env.USE_NEON
```

```
$ ck compile program:image-classification-armnn-tflite --env.USE_OPENCL
```

```
$ ck compile program:image-classification-armnn-tflite --env.USE_NEON --env.USE_OPENCL
```

## Run (assuming the same options for the backend)

```
$ ck run program:image-classification-armnn-tflite  --env.CK_BATCH_COUNT=5 --env.USE_NEON
```
**Here:**
 - CK_BATCH_COUNT - file's count to evaluate (default: 1)
 - USE_NEON - asking for Cpu acceleration backend support
 - USE_OPENCL - asking for Gpu acceleration backend support
