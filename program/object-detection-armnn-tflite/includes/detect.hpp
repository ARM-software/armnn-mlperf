/*
 * Copyright (c) 2019 dividiti Ltd.
 * Copyright (c) 2019 Arm Ltd.
 *
 * SPDX-License-Identifier: MIT.
 */

#ifndef DETECT_H
#define DETECT_H

#include <iomanip>
#include <vector>
#include <iterator>

#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

#define OBJECT_DETECTION_ARMNN_TFLITE
#include "settings.h"
#include "non_max_suppression.h"
#include "benchmark.h"

using namespace std;
using namespace CK;

template <typename TData, typename TInConverter, typename TOutConverter>
class ArmNNBenchmark : public Benchmark<TData, TInConverter, TOutConverter> {
public:
    ArmNNBenchmark(Settings* settings,
                   TData *in_ptr,
                   TData *boxes_ptr,
                   TData *scores_ptr
                   )
            : Benchmark<TData, TInConverter, TOutConverter>(settings, in_ptr, boxes_ptr, scores_ptr) {
    }
};

armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
        armnn::TensorInfo>& input, const void* inputTensorData)
{
    return { {input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
        armnn::TensorInfo>& output, void* outputTensorData)
{
    return { {output.first, armnn::Tensor(output.second, outputTensorData) } };
}

void AddTensorToOutput(armnn::OutputTensors &v, const std::pair<armnn::LayerBindingId,
        armnn::TensorInfo>& output, void* outputTensorData ) {
    v.push_back({output.first, armnn::Tensor(output.second, outputTensorData) });
}

#endif //DETECT_H
