#include "ITask.h"
#include "../../Utils/visionOps.h"


class IDetection : public ITask {
private:
	

public :
    using InputType = cv::Mat;
    using OutputType = std::vector<std::vector<float>>;	// [x, y, width, height, confidence]
    static void preProcess(const InputType& before, cv::Mat& after) {
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
    static void preProcess(const InputType& before, float* deviceBuffer) {
        // Resize the image to match the model's input size.
        cv::Mat resized;
        cv::resize(before, resized, cv::Size(640, 640));

        // Convert the image to float and normalize it (e.g., scale to [0, 1]).
        cv::Mat normalized;
        resized.convertTo(normalized, CV_32FC3, 1.0 / 255);

        // Convert HWC (Height x Width x Channels) to CHW (Channels x Height x Width).
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);
        float* hostData = new float[3 * 640 * 640];
        for (int c = 0; c < 3; ++c) {
            memcpy(hostData + c * 640 * 640, channels[c].data, 640 * 640 * sizeof(float));
        }

        // Copy the preprocessed data to the GPU.
        cudaMemcpy(deviceBuffer, hostData, 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice);

        delete[] hostData;
    }


    /**
    *@todo postProcess function that uses custom nms(iou comparing) -> not working correctly.
    * 
    */
    /*
    static void postProcess(const torch::Tensor& tensor, OutputType& output, float confThreshold = 0.5, float nmsThreshold = 0.4) {
        try {
            //yolo11 -> tensor with shape [1,84,8400]

            output.clear();

            // Tensor to CPU and detach
            auto rawOutput = tensor.to(torch::kCPU).detach();

            // Access dimensions (assumes YOLO's output format)
            auto numBoxes = rawOutput.size(2);

            // Extract bounding boxes and class scores
            auto bboxes = tensor.index({ 0, torch::indexing::Slice(0, 4), torch::indexing::Ellipsis }); // Shape: [4, 8400]
            auto class_scores = tensor.index({0, torch::indexing::Slice(4, torch::indexing::None), torch::indexing::Ellipsis }); // Shape: [80, 8400]

            // Compute maximum class score and class indices for each box
            auto max_scores = std::get<0>(class_scores.max(0)); // Shape: [8400]
            std::cout << max_scores.size(0) << std::endl;
            auto class_indices = std::get<1>(class_scores.max(0)); // Shape: [8400]
            std::cout << class_indices.size(0) << std::endl;

            // Apply confidence threshold
            auto mask = (max_scores > confThreshold); // Shape: [8400]
            std::cout << mask.size(0) << std::endl;

            // Filter bounding boxes
            auto filtered_bboxes = bboxes.index({ torch::indexing::Slice(), mask }); // Shape: [4, N]
            auto filtered_scores = max_scores.index({ mask }); // Shape: [N]
            auto filtered_classes = class_indices.index({ mask }); // Shape: [N]

            // Transpose filtered_bboxes for NMS
            auto bboxes_for_nms = filtered_bboxes.t(); // Shape: [N, 4]
            
            // Apply NMS to reduce redundant boxes
            auto kept_indices = visionOps::nms(bboxes_for_nms, filtered_scores, nmsThreshold);


            // Filter output to keep only NMS results
            OutputType filteredOutput;

            for (int i = 0; i < kept_indices.size(0); ++i) {
                int64_t idx = kept_indices[i].item<int64_t>(); // Get index for each kept box

                // Extract the bounding box coordinates (x, y, w, h)
                auto bbox = bboxes_for_nms[idx];
                std::vector<float> bbox_vec = { bbox[0].item<float>(), bbox[1].item<float>(), bbox[2].item<float>(), bbox[3].item<float>() };

                // Assuming predicted_classes is a tensor with class IDs
                float class_id = filtered_classes[idx].item<float>(); // Get the class ID
                float confidence = filtered_scores[idx].item<float>(); // Get the confidence score

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
    */

    static void postProcess(const torch::Tensor& tensor, OutputType& output, float confThreshold = 0.5, float nmsThreshold = 0.4) {
        try {
            // Clear the output
            output.clear();

            // Move tensor to CPU and detach
            auto rawOutput = tensor.to(torch::kCPU).detach();

            // Access dimensions (assumes YOLO's output format)
            // Shape : [1,4,8400]
            auto numBoxes = rawOutput.size(2);

            // Extract bounding boxes and class scores
            auto bboxes = tensor.index({ 0, torch::indexing::Slice(0, 4), torch::indexing::Ellipsis }); // Shape: [4, 8400]
            auto class_scores = tensor.index({ 0, torch::indexing::Slice(4, torch::indexing::None), torch::indexing::Ellipsis }); // Shape: [80, 8400]

            // Compute maximum class scores and their indices for each box
            auto max_scores = std::get<0>(class_scores.max(0)); // Shape: [8400]
            auto class_indices = std::get<1>(class_scores.max(0)); // Shape: [8400]

            // Apply confidence threshold
            auto mask = (max_scores > confThreshold); // Shape: [8400]

            // Filter bounding boxes and scores
            auto filtered_bboxes = bboxes.index({ torch::indexing::Slice(), mask }); // Shape: [4, N]
            auto filtered_scores = max_scores.index({ mask }); // Shape: [N]
            auto filtered_classes = class_indices.index({ mask }); // Shape: [N]

            // Convert filtered bounding boxes to OpenCV format
            std::vector<cv::Rect> cv_bboxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;

            for (int i = 0; i < filtered_bboxes.size(1); ++i) {
                auto bbox = filtered_bboxes.index({ torch::indexing::Slice(), i });
                float x = bbox[0].item<float>();
                float y = bbox[1].item<float>();
                float w = bbox[2].item<float>();
                float h = bbox[3].item<float>();

                cv_bboxes.emplace_back(cv::Rect(cv::Point(x - w / 2, y - h / 2), cv::Size(w, h))); // Convert to OpenCV Rect format
                confidences.push_back(filtered_scores[i].item<float>());
                class_ids.push_back(filtered_classes[i].item<int>());
            }

            // Apply OpenCV NMS
            std::vector<int> kept_indices;
            cv::dnn::NMSBoxes(cv_bboxes, confidences, confThreshold, nmsThreshold, kept_indices);

            // Filter output to keep only NMS results
            OutputType filteredOutput;
            for (const auto& idx : kept_indices) {
                std::vector<float> bbox_vec = {
                    static_cast<float>(cv_bboxes[idx].x),
                    static_cast<float>(cv_bboxes[idx].y),
                    static_cast<float>(cv_bboxes[idx].width),
                    static_cast<float>(cv_bboxes[idx].height),
                    static_cast<float>(class_ids[idx]),
                    confidences[idx]
                };
                filteredOutput.push_back(bbox_vec);
            }

            output = std::move(filteredOutput);
        }
        catch (std::exception& ex) {
            std::cout << "Postprocessing Error: " << ex.what() << std::endl;
        }
    }

    static void postProcess(const float* deviceOutput, OutputType& output,
        float confThreshold = 0.5, float nmsThreshold = 0.4) {
        try {
            // Define the size of the output tensor
            const int outputSize = 84 * 8400; // Example size, adjust based on your model

            // Wrap the device pointer in a torch::Tensor (GPU tensor)
            auto tensor = torch::from_blob(const_cast<float*>(deviceOutput),
                { 1, 84, 8400 }, // Adjust dimensions as per your model
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            // Move tensor to CPU if needed (TensorRT outputs are usually on GPU)
            auto cpuTensor = tensor.to(torch::kCPU).clone(); // Clone to avoid pointer aliasing

            // Call your existing postProcess function
            postProcess(cpuTensor, output, confThreshold, nmsThreshold);
        }
        catch (std::exception& ex) {
            std::cerr << "Postprocessing Error: " << ex.what() << std::endl;
        }
    }
};