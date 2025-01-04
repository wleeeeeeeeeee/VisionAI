#include "ITask.h"


/**
* @brief A Base Class for Segmentation. Provide general segmentation pre,post process ops.
*/
class ISegmentation : public ITask {
private :

protected:
    /*
    * Normalization
    * Resizing
    */
public:
    using InputType = cv::Mat;	// opencv Image
    using OutputType = std::vector<cv::Mat>;

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
    static void postProcess(std::vector<torch::jit::IValue>& rawOuputs, OutputType& outputvec) {
        try {
            for (int i = 0; i < rawOuputs.size(); i++) {
                auto shape = rawOuputs[i].toTensor().sizes();
                for (auto dim : shape) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
            }

            float* data0_ptr = new float[1 * 116 * 8400];
            auto output0 = rawOuputs[0].toTensor().to(torch::kFloat).to(torch::kCPU);
            float* data1_ptr = new float[1 * 32 * 160 * 160];
            auto output1 = rawOuputs[1].toTensor().to(torch::kFloat).to(torch::kCPU);

            std::copy(output0.data_ptr<float>(), output0.data_ptr<float>() + 1 * 116 * 8400, data0_ptr);
            std::copy(output1.data_ptr<float>(), output1.data_ptr<float>() + 1 * 32 * 160 * 160, data1_ptr);

            postProcess(data0_ptr, data1_ptr, outputvec);

            delete[] data0_ptr;
            delete[] data1_ptr;
        }
        catch (std::exception& ex) {
            std::cerr << "During Segmentation PostProcessing, error occured : " << ex.what() << std::endl;
        }
    }
    static void postProcess(float* data0_ptr, float* data1_ptr, OutputType& outputvec, float scoreThreshold = 0.2, float nmsThreshold = 0.4) {
        try {
            
       }
        catch (std::exception& ex) {
            std::cout << "Error Occured During Segmentation PostProcess : " << ex.what()<< std::endl;
        }
    }
};
