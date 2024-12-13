#pragma once
#include <vector>
/// include opencv
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