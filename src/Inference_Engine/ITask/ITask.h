#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <NvInfer.h>

/// include torch
/// utils -> data transformation , augementation, feature visualization, output saving, etc.
/// 
/// 
class ITask {
private:
public:
	virtual void infer();
	virtual ~ITask() = default;
	static void LetterBox(cv::Mat& input_image, cv::Mat& output_image, cv::Vec4d& params, cv::Size shape = cv::Size(640, 640), cv::Scalar color = cv::Scalar(114, 114, 114))
	{
		float r = std::min((float)shape.height / (float)input_image.rows, (float)shape.width / (float)input_image.cols);
		float ratio[2]{ r, r };
		int new_un_pad[2] = { (int)std::round((float)input_image.cols * r),(int)std::round((float)input_image.rows * r) };

		auto dw = (float)(shape.width - new_un_pad[0]) / 2;
		auto dh = (float)(shape.height - new_un_pad[1]) / 2;

		if (input_image.cols != new_un_pad[0] && input_image.rows != new_un_pad[1])
			cv::resize(input_image, output_image, cv::Size(new_un_pad[0], new_un_pad[1]));
		else
			output_image = input_image.clone();

		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));

		params[0] = ratio[0];
		params[1] = ratio[1];
		params[2] = left;
		params[3] = top;

		cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
	}
};