#pragma once

/// include opencv
/// include torch
/// utils -> data transformation , augementation, feature visualization, output saving, etc.
/// 

template <class Model>
class Detector {
private:
	Model model;

public :
	Detector();
	virtual void Initialize();
	virtual void Train();
	virtual void LoadWeight();
	virtual void LoadPretrained();
	virtual void Predict();
};