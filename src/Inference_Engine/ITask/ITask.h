#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

/// include torch
/// utils -> data transformation , augementation, feature visualization, output saving, etc.
/// 
/// 
class ITask {
private:
public:
	virtual void infer();
	virtual ~ITask() = default;
};