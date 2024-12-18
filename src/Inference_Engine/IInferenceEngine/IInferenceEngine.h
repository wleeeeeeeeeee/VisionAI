#pragma once
#include <iostream>
#include <onnxruntime_cxx_api.h>

template <typename TaskType>
class IInferenceEngine {

private :
	
public:
	
	virtual void loadModel(const std::string& modelPath) = 0;
	virtual void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) = 0;
	virtual ~IInferenceEngine() = default;
};