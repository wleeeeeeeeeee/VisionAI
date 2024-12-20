#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class dataTransformer {
private:
public:
	static void mat2tensor(const cv::Mat& mat, at::Tensor& tensor) {
		try {
			// Convert cv::Mat to float and normalize to [0, 1]
			cv::Mat floatMat;
			mat.convertTo(floatMat, CV_32F, 1.0 / 255);

			// Convert cv::Mat to at::Tensor (CHW format required for most AI models)
			tensor = torch::from_blob(floatMat.data, { mat.rows, mat.cols, mat.channels() }, torch::kFloat);
			tensor = tensor.permute({ 2, 0, 1 }); // Convert to CHW layout
		}
		catch (std::exception& ex) {
			//some kind of error logging
		}
	}
	static void tensor2mat(const at::Tensor& tensor, cv::Mat& mat) {
		try {
			// Ensure tensor is contiguous
			at::Tensor contiguousTensor = tensor.contiguous();

			// Convert at::Tensor to cv::Mat (HWC layout)
			auto sizes = contiguousTensor.sizes();
			int height = sizes[1];
			int width = sizes[2];
			int channels = sizes[0];

			// Create cv::Mat from tensor data
			mat = cv::Mat(cv::Size(width, height), CV_32FC(channels), contiguousTensor.data_ptr<float>()).clone();

			// Convert to 8-bit format (if needed)
			mat.convertTo(mat, CV_8U, 255.0);
		}
		catch (std::exception& ex) {
			//some kind of error logging
		}
	}

	/**
	* @todo need to work on transforming cv::mat, at::Tensor data to std::vector
	*/
	static void mat2vec(const cv::Mat& mat, std::vector<float>& vec) {
		try {
			// Convert cv::Mat to float (if not already)
			cv::Mat floatMat;
			mat.convertTo(floatMat, CV_32F, 1.0 / 255);

			// Copy data to std::vector
			vec.assign((float*)floatMat.data, (float*)floatMat.data + floatMat.total() * floatMat.channels());
		}
		catch (const std::exception& ex) {
			std::cout << "Error occured in mat2vec : " << ex.what() << std::endl;
		}
	}
	static void tensor2vec(const at::Tensor& tensor, std::vector<float>& vec) {
		try {
			// Ensure tensor is contiguous
			at::Tensor contiguousTensor = tensor.contiguous();

			// Copy data to std::vector
			vec.assign(contiguousTensor.data_ptr<float>(),
				contiguousTensor.data_ptr<float>() + contiguousTensor.numel());
		}
		catch (const std::exception& ex) {
			std::cout << "Error occured in tensor2vec : " << ex.what() << std::endl;
		}
	}

	// Converts std::vector<float> to at::Tensor
	static void vec2tensor(const std::vector<float>& vec, const std::vector<int64_t>& shape, at::Tensor& tensor) {
		try {
			tensor = torch::from_blob(const_cast<float*>(vec.data()), shape, torch::kFloat).clone();
		}
		catch (const std::exception& ex) {
			std::cout << "Error occured in vec2tensor : " << ex.what() << std::endl;
		}
	}

	// Converts std::vector<float> to cv::Mat
	static void vec2mat(const std::vector<float>& vec, int rows, int cols, int channels, cv::Mat& mat) {
		try {
			mat = cv::Mat(rows, cols, CV_32FC(channels), const_cast<float*>(vec.data())).clone();
			mat.convertTo(mat, CV_8U, 255.0); // Convert to 8-bit
		}
		catch (const std::exception& ex) {
			std::cout << "Error occured in vec2mat : " << ex.what() << std::endl;
		}
	}

};