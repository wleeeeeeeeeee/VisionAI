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

	void checkInputsDevice(const std::vector<torch::jit::IValue>& inputs, torch::DeviceType expectedDevice) {
		for (size_t i = 0; i < inputs.size(); ++i) {
			if (inputs[i].isTensor()) {
				torch::Tensor tensor = inputs[i].toTensor();
				if (tensor.device().type() != expectedDevice) {
					std::cerr << "Input tensor at index " << i << " is on device "
						<< tensor.device() << ", but expected " << expectedDevice << std::endl;
				}
				else {
					std::cout << "Input tensor at index " << i << " is correctly on "
						<< (expectedDevice == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
				}
			}
			else {
				std::cerr << "Input at index " << i << " is not a tensor!" << std::endl;
			}
		}
	}

public:
	TorchInferenceEngine(const std::string& modelPath) {
		if (!torch::cuda::is_available()) {
			std::cerr << "CUDA is not available. Setting device to CPU." << std::endl;
			this->device = at::kCPU;
		}
		else {
			std::cerr << "CUDA is available. Using CUDA as default device." << std::endl;
			//this->device = at::kCPU;
			this->device = at::kCUDA;
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
		input_tensor = input_tensor.to(device);
		this->inputs.clear();
		this->inputs.push_back(torch::jit::IValue(input_tensor));

		//run inference
		auto forward_output = this->model.forward(this->inputs);
		this->outputs.clear();
		//mutiple outputs
		if (forward_output.isTuple()) {
			auto output_tuple = forward_output.toTuple();
			for (const auto& element : output_tuple->elements()) {
				this->outputs.push_back(element);
			}
		}
		else {
			this->outputs.push_back(forward_output);
		}

		//float* temp0 = this->outputs[0].toTensor().to(torch::kFloat).to(at::kCPU).data_ptr<float>();
		//float* temp1 = this->outputs[1].toTensor().to(torch::kFloat).to(at::kCPU).data_ptr<float>();

		//task-specific postprocessing
		TaskType::postProcess(this->outputs, output);
		
	}
};