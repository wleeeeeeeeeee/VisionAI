#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"
#include "../../Log/trtLogger.h"


template <typename TaskType>

class TensorRTInferenceEngine : public IInferenceEngine<TaskType> {
private:
	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* execution_context;
	cudaStream_t stream;
	float* input_device;
	float* input_host;
	float* output_device;
	float* output_host;
	float* bindings[2];

public:
	TensorRTInferenceEngine(const std::string& modelPath) {
		loadModel(modelPath);
	}

	void loadModel(const std::string& modelPath) override {
		try {
			if (!std::filesystem::exists(modelPath)) {
				std::cerr << "Engine does not exists !" << std::endl;
				std::exit(-1);
			}
			std::ifstream file(modelPath, std::ios::binary);
			if (!file.good()) {
				std::cerr << "Cannot read model" << std::endl;
				std::exit(-1);
			}
			std::stringstream buffer;
			buffer << file.rdbuf();

			std::string stream_model(buffer.str());

			TensorRTLogger Logger;
			this->runtime = nvinfer1::createInferRuntime(Logger);
			this->engine = this->runtime->deserializeCudaEngine(stream_model.data(), stream_model.size());

			cudaStreamCreate(&this->stream);
			this->execution_context = this->engine->createExecutionContext();

			int input_size = 1 * 640 * 640 * 3;
			int output_size = 1 * 84 * 8400;
			cudaMallocHost(reinterpret_cast<void**>(& this->input_host), sizeof(float) * input_size);
			cudaMallocHost(reinterpret_cast<void**>(& this->output_host), sizeof(float) * output_size);

			cudaMalloc(reinterpret_cast<void**>(& this->input_device), sizeof(float) * input_size);
			cudaMalloc(reinterpret_cast<void**>(&this->output_device), sizeof(float)* output_size);

			this->bindings[0] = this->input_device;
			this->bindings[1] = this->output_device;

		}
		catch (std::exception& ex) {
			std::cerr << ex.what() << std::endl;
		}
		
	}

	void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) override {
		try {
		//prepare input tensor
		TaskType::preProcess(input, this->input_device);
		//run inference
		this->execution_context->executeV2((void**)bindings);
		//task-specific postprocessing
		TaskType::postProcess(this->output_device, output);
		}
		catch (std::exception& ex) {
			std::cerr << ex.what() << std::endl;
		}
	}
};