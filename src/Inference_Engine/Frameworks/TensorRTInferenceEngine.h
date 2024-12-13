#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"


template <typename TaskType>

class TensorRTInferenceEngine : public IInferenceEngine<TaskType> {
private:
	// torch::jit::scipt::Module model;
	// ort::session session;
	// nvrt::nvinfer engine;
public:
	void loadModel(const std::string& modelPath) override {

		//model = torch::jit::load(modelPath);
		//session = 
		// engine = 
	}

	void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) override {
		//prepare input tensor

		//run inference

		//task-specific postprocessing
		TaskType::processOuput(rawOutput, output);
	}
};