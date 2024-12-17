#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class dataTransformer {
private:
public:
	static void mat2tensor(const cv::Mat& mat, at::Tensor& tensor) {
		try {

		}
		catch (std::exception& ex) {
			//some kind of error logging
		}
	}
	static void tensor2mat(const at::Tensor& tensor, cv::Mat& mat) {
		try {

		}
		catch (std::exception& ex) {
			//some kind of error logging
		}
	}

	/**
	* @todo need to work on transforming cv::mat, at::Tensor data to std::vector
	*/
	static void mat2vec() {

	}
	static void tensor2vec() {

	}
};