/*
 * Copyright (c) 2019 dividiti Ltd.
 * Copyright (c) 2019 Arm Ltd.
 *
 * SPDX-License-Identifier: MIT.
 */
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include <sys/stat.h>
#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include "benchmark.h"

using namespace std;
using namespace CK;

template <typename TData, typename TInConverter, typename TOutConverter>
class ArmNNBenchmark : public Benchmark<TData, TInConverter, TOutConverter> {
public:
    ArmNNBenchmark(const BenchmarkSettings* settings, TData *in_ptr, TData *out_ptr, int input_index)
            : Benchmark<TData, TInConverter, TOutConverter>(settings, in_ptr, out_ptr) {
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


int main(int argc, char* argv[]) {

    try {
        bool use_neon                   = getenv_b("USE_NEON");
        bool use_opencl                 = getenv_b("USE_OPENCL");
        string input_layer_name         = getenv_s("CK_ENV_TENSORFLOW_MODEL_INPUT_LAYER_NAME");
        string output_layer_name        = getenv_s("CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME");

        init_benchmark();

        BenchmarkSettings settings;

        // TODO: learn how to process batches
        // currently interpreter->tensor(input_index)->dims[0] = 1
        if (settings.batch_size != 1)
            throw string("Only BATCH_SIZE=1 is currently supported");

        BenchmarkSession session(&settings);

        unique_ptr<IBenchmark> benchmark;
        armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
        armnn::NetworkId networkIdentifier;
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
        armnn::OutputTensors outputTensor;
        armnn::InputTensors inputTensor;

        // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
        //std::vector<armnn::BackendId> optOptions = {armnn::Compute::CpuAcc, armnn::Compute::GpuAcc};
        std::vector<armnn::BackendId> optOptions = {armnn::Compute::CpuRef};
        if( use_neon && use_opencl) {
            optOptions = {armnn::Compute::CpuAcc, armnn::Compute::GpuAcc};
        } else if( use_neon ) {
            optOptions = {armnn::Compute::CpuAcc};
        } else if( use_opencl ) {
            optOptions = {armnn::Compute::GpuAcc};
        }

        cout << "\nLoading graph..." << endl;
        measure_setup([&]{
            armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(settings.graph_file.c_str());
            if (!network)
                throw "Failed to load graph from file";

            armnnTfLiteParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo(0, input_layer_name);
            armnnTfLiteParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, output_layer_name);

            armnn::TensorShape inShape = inputBindingInfo.second.GetShape();
            armnn::TensorShape outShape = outputBindingInfo.second.GetShape();
            std::size_t inSize = inShape[0] * inShape[1] * inShape[2] * inShape[3];
            std::size_t outSize = outShape[0] * outShape[1];

            armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, optOptions, runtime->GetDeviceSpec());

            runtime->LoadNetwork(networkIdentifier, std::move(optNet));

            armnn::DataType input_type = inputBindingInfo.second.GetDataType();
            armnn::DataType output_type = outputBindingInfo.second.GetDataType();
            if (input_type != output_type)
                throw format("Type of graph's input (%d) does not match type of its output (%d).", int(input_type), int(output_type));

            void* input = input_type == armnn::DataType::Float32 ? (void*)new float[inSize] : (void*)new uint8_t[inSize];
            void* output = output_type == armnn::DataType::Float32 ? (void*)new float[outSize] : (void*)new uint8_t[outSize];
 
            inputTensor = MakeInputTensors(inputBindingInfo, input);
            outputTensor = MakeOutputTensors(outputBindingInfo, output);

            switch (input_type) {
                case armnn::DataType::Float32:
                    if (settings.skip_internal_preprocessing)
                        benchmark.reset(new ArmNNBenchmark<float, InCopy, OutCopy>(&settings, (float*)input, (float*)output, 0));
                    else
                        benchmark.reset(new ArmNNBenchmark<float, InNormalize, OutCopy>(&settings, (float*)input, (float*)output, 0));
                    break;

                case armnn::DataType::QuantisedAsymm8:
                    benchmark.reset(new ArmNNBenchmark<uint8_t, InCopy, OutDequantize>(&settings, (uint8_t*)input, (uint8_t*)output, 0));
                    break;

                default:
                    throw format("Unsupported type of graph's input: %d. "
                                 "Supported types are: Float32 (%d), UInt8 (%d)",
                                 int(input_type), int(armnn::DataType::Float32), int(armnn::DataType::QuantisedAsymm8));
            }

            int out_num = outShape[0];
            int out_classes = outShape[1];
            cout << format("Output tensor dimensions: %d*%d", out_num, out_classes) << endl;
            if (out_classes != settings.num_classes && out_classes != settings.num_classes+1)
                throw format("Unsupported number of classes in graph's output tensor. Supported numbers are %d and %d",
                             settings.num_classes, settings.num_classes+1);
            benchmark->has_background_class = out_classes == settings.num_classes+1;
        });

        cout << "\nProcessing batches..." << endl;
        measure_prediction([&]{
            while (session.get_next_batch()) {
                session.measure_begin();
                benchmark->load_images(session.batch_files());
                session.measure_end_load_images();

                session.measure_begin();
                if (runtime->EnqueueWorkload(networkIdentifier, inputTensor, outputTensor) != armnn::Status::Success)
                    throw "Failed to invoke the classifier";
                session.measure_end_prediction();

                benchmark->save_results(session.batch_files());
            }
        });

        finish_benchmark(session);
    }
    catch (const string& error_message) {
        cerr << "ERROR: " << error_message << endl;
        return -1;
    }
    return 0;
}
