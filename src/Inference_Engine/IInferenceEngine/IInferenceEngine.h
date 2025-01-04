#pragma once
#include <iostream>
#include <filesystem>

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>



/**
* @brief A base inference engine serve as an abstract class to support various deep learning frameworks.
* 
*/
template <typename TaskType>
class IInferenceEngine {

private :
	//Model
	TaskType Model;
	//Model Shape

	//Model Path

	//Device Type

protected:
public:
	/*
	virtual void loadModel();
	virtual void readModel();
	virtual void warmUp();
	virtual void forward();
	virtual void infer();
	*/

	//@deprecate
	virtual void loadModel(const std::string& modelPath) = 0;
	virtual void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) = 0;
	virtual ~IInferenceEngine() = default;
};