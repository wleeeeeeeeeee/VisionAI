#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"

template <typename TaskType>
class OnnxInferenceEngine : public IInferenceEngine<TaskType> {
private:
	Ort::Env env{};
	Ort::AllocatorWithDefaultOptions allocator{};
	Ort::RunOptions runOptions{};
	Ort::Session session = Ort::Session(nullptr);
public:
	void loadModel(const std::string& modelPath) override {
		//load model into session
		//configure gpu, cpu, any necessary inference options
		//get onnx info (node names, input shape, output shape etc)
	}

	void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) override {
		//prepare input tensor

		//run inference

		//task-specific postprocessing
		TaskType::processOuput(rawOutput, output);
	}
};