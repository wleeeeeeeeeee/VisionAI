#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <NvInfer.h>


/**
* @brief A base class where every tasks and models will derive from.
* @details Basic task operations will be in task (iclassification, idetection, isegmentation, etc). Custom Models need to have their own custom pre,postProcesses to override infer();
*/
class ITask {
private:
public:
	/*
	*  virtual void preProcess();
	*  virtual void postProcess();
	*/
	virtual ~ITask() = default;
};