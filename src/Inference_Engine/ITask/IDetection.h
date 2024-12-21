#include "ITask.h"
#include "../../Utils/visionOps.h"


class IDetection : public ITask {
private:
	

public :
    using InputType = cv::Mat;
    using OutputType = std::vector<std::vector<float>>;	// [x, y, width, height, confidence]
    static void preProcess(const cv::Mat& before, cv::Mat& after) {
        try {
            //resizing to yolo inputsize
            cv::resize(before, after, cv::Size(640, 640));

            //normalizing pixel values
            after.convertTo(after, CV_32FC3, 1.0 / 255.0);

            //converting bgr to rgb
            cv::cvtColor(after, after, cv::COLOR_BGR2RGB);

        }
        catch (std::exception& ex) {
            std::cout << ex.what() << std::endl;
        }
    }
    static void preProcess(const cv::Mat& before, at::Tensor& after) {
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

    static void postProcess(const torch::Tensor& tensor, OutputType& output, float confThreshold = 0.5, float nmsThreshold = 0.4) {
        try {
            //yolo11 -> tensor with shape [1,84,8400]

            output.clear();

            // Tensor to CPU and detach
            auto rawOutput = tensor.to(torch::kCPU).detach();

            // Access dimensions (assumes YOLO's output format)
            auto numBoxes = rawOutput.size(2);
            int num_classes = rawOutput.size(1) - 5; 

            //x,y,w,h, conf
            torch::Tensor objectness_scores = tensor.index({ 0,4 });                     // shape : [8400]
            torch::Tensor class_scores = tensor.index({ 0,torch::indexing::Slice(5) });  // shape : [80.8400]


            auto mask = (objectness_scores > confThreshold); //marking 0:false 1:true [80]
            auto indices = torch::nonzero(mask).squeeze(1);
            torch::Tensor valid_objectness = objectness_scores.index({ indices });
            torch::Tensor valid_class_scores = class_scores.index({ torch::indexing::Ellipsis , indices });
            torch::Tensor valid_bboxes = tensor.index({ 0,torch::indexing::Slice(0,4), indices });

            torch::Tensor adjusted_scores = valid_class_scores * valid_objectness;
            auto class_probs = adjusted_scores.max(0);
            torch::Tensor class_confidences = std::get<0>(class_probs);
            torch::Tensor predicted_classes = std::get<1>(class_probs);

            valid_bboxes = valid_bboxes.t();
            
            // Apply NMS to reduce redundant boxes
            double iou_threshold = 0.5;
            auto kept_indices = visionOps::nms(valid_bboxes, class_confidences, iou_threshold);

            // Filter output to keep only NMS results
            OutputType filteredOutput;

            for (int i = 0; i < kept_indices.size(0); ++i) {
                int64_t idx = kept_indices[i].item<int64_t>(); // Get index for each kept box

                // Extract the bounding box coordinates (x, y, w, h)
                auto bbox = valid_bboxes.index({ idx , torch::indexing::Ellipsis, }); // [x, y, w, h]
                std::vector<float> bbox_vec = { bbox[0].item<float>(), bbox[1].item<float>(), bbox[2].item<float>(), bbox[3].item<float>() };

                // Assuming predicted_classes is a tensor with class IDs
                float class_id = predicted_classes[idx].item<float>(); // Get the class ID
                float confidence = class_confidences[idx].item<float>(); // Get the confidence score

                bbox_vec.push_back(class_id);
                bbox_vec.push_back(confidence);

                filteredOutput.push_back(bbox_vec); // Add filtered bounding box to the output
            }


            output = std::move(filteredOutput);
        }
        catch (std::exception& ex) {
            std::cout << "Postprocessing Error: " << ex.what() << std::endl;
        }
    }
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