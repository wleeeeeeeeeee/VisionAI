#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"

template <typename TaskType>

class TorchInferenceEngine : public IInferenceEngine<TaskType> {
private:
	torch::DeviceType device;
	torch::jit::script::Module model;
	std::vector<torch::jit::IValue> input;
	//torch::jit::IValue input;
	torch::jit::IValue output;

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
		//define device
		//device = at::kCUDA;
		this->device = at::kCPU;
		//mode model to device
		model.to(device);

	}

	void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) override {
		//prepare input tensor
		torch::Tensor input_tensor;
		TaskType::preProcess(input, input_tensor);
		this->input.push_back(input_tensor);
		
		//run inference
		this->output = this->model.forward(this->input);
		//task-specific postprocessing
		TaskType::postProcess(this->output.toTensor(), output);
	}
};