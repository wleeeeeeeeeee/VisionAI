#pragma once
#include "ITask.h"


class IDetection : public ITask {
private:
	

public :
    using InputType = std::vector<float>;	// Flattened image data
    using OutputType = std::vector<std::vector<float>>;	// [x, y, width, height, confidence]
    static void processOutput(const torch::Tensor& tensor, OutputType& output) {
        output.clear();
        for (int i = 0; i < tensor.size(0); ++i) {
            output.push_back({
                tensor[i][0].item<float>(),  // x
                tensor[i][1].item<float>(),  // y
                tensor[i][2].item<float>(),  // width
                tensor[i][3].item<float>(),  // height
                tensor[i][4].item<float>()   // confidence
                });
        }
    }
};