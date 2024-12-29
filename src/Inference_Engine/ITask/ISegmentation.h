#include "ITask.h"

class ISegmentation : public ITask {
protected:
    int cls_idx = 0;    //background class is chosen for default
public:
    using InputType = cv::Mat;	// opencv Image
    using OutputType = std::vector<std::vector<float>>;	
    static void preProcess(const InputType& before, at::Tensor& after) {
        try {

            cv::Mat resized;
            //resizing to yolo inputsize
            cv::resize(before, resized, cv::Size(640, 640));

            //normalizing pixel values
            resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

            //converting bgr to rgb
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

            //convert to tensor
            auto tensor = torch::from_blob(resized.data, { 1,resized.rows, resized.cols, 3 }, torch::kFloat32);

            //permute to match NCHW format
            tensor = tensor.permute({ 0,3,1,2 });

            //save processed tensor for inference
            after = tensor.clone();

        }
        catch (std::exception& ex) {
            std::cout << ex.what() << std::endl;
        }
    }
    static void postProcess(std::vector<torch::jit::IValue>& rawOuputs, OutputType& output) {
        try {

       }
        catch (std::exception& ex) {
            std::cout << "Error Occured During Segmentation PostProcess" << std::endl;
        }
    }
};
