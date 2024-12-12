#pragma once

/// include opencv
/// include torch
/// utils -> data transformation , augementation, feature visualization, output saving, etc.
/// 


class Detector {
public :
	Detector();
	virtual void Initialize();
	virtual void Train();
	virtual void LoadWeight();
	virtual void LoadPretrained();
	virtual void Predict();


};