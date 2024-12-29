#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"

template <typename TaskType>

class TorchInferenceEngine : public IInferenceEngine<TaskType> {
private:
	torch::DeviceType device;
	torch::jit::script::Module model;
	std::vector<torch::jit::IValue> inputs;
	//torch::jit::IValue input;
	std::vector<torch::jit::IValue> outputs;

public:
	TorchInferenceEngine(const std::string& modelPath) {
		if (!torch::cuda::is_available()) {
			std::cerr << "CUDA is not available. Setting device to CPU." << std::endl;
			this->device = at::kCPU;
		}
		else {
			std::cerr << "CUDA is available. Using CUDA as default device." << std::endl;
			this->device = torch::kCUDA;
		}

		loadModel(modelPath);
	}
	void loadModel(const std::string& modelPath) override {

		//checking file path
		if (!std::filesystem::exists(modelPath)) {
			std::cerr << "model does not exist !" << std::endl;
			std::exit(1);
		}
		//load model
		model = torch::jit::load(modelPath);
		//mode model to device
		model.to(device);

	}

	void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) override {
		//prepare input tensor
		torch::Tensor input_tensor;
		TaskType::preProcess(input, input_tensor);
		this->inputs.push_back(input_tensor);
		
		//run inference
		this->outputs.push_back(this->model.forward(this->input));
		//task-specific postprocessing
		TaskType::postProcess(this->output, output);
	}
};