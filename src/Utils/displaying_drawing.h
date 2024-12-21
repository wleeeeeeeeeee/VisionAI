#pragma once

#include <fstream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class displaying_drawing {
private:
public:
    static const std::vector<std::string>& loadClassNames(const std::string& file_path) {
        static std::vector<std::string> class_names; // Static to persist the data
        class_names.clear(); // Clear previous data if any

        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + file_path);
        }

        std::string line;
        while (std::getline(file, line)) {
            class_names.push_back(line);
        }

        file.close();
        return class_names;
    }


	static void drawBoundingBoxes(cv::Mat& image, const std::vector<std::vector<float>> bboxes,
        const std::vector<std::string>& class_names, int resized_width, int resized_height) {
        int original_width = image.cols;
        int original_height = image.rows;

        // Compute scaling factors
        float scale_x = static_cast<float>(original_width) / resized_width;
        float scale_y = static_cast<float>(original_height) / resized_height;

        for (int i = 0; i < bboxes.size(); i++) {
            // Get bounding box coordinates
            auto box = bboxes[i]; // Shape: [4]
            //float x_min = box[0] * scale_x;
            //float y_min = box[1] * scale_y;
            //float x_max = box[2] * scale_x;
            //float y_max = box[3] * scale_y;


            float center_x = box[0] * scale_x;
            float center_y = box[1] * scale_y;
            float width = box[2] * scale_x;
            float height = box[3] * scale_y;

            float x_min = center_x - width / 2.0;
            float y_min = center_y - height / 2.0;
            float x_max = center_x + width / 2.0;
            float y_max = center_y + height / 2.0;





            int class_id = box[4];
            float confidence = box[5];

            // Draw rectangle on the original image
            cv::rectangle(image, cv::Point(static_cast<int>(x_min), static_cast<int>(y_min)),
                cv::Point(static_cast<int>(x_max), static_cast<int>(y_max)), cv::Scalar(0, 255, 0), 2);

            // Add label
            std::string label = class_names[class_id] + " (" + std::to_string(confidence) + ")";
            int base_line = 0;
            auto label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
            cv::rectangle(image, cv::Point(static_cast<int>(x_min), static_cast<int>(y_min) - label_size.height - 10),
                cv::Point(static_cast<int>(x_min) + label_size.width, static_cast<int>(y_min)), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(image, label, cv::Point(static_cast<int>(x_min), static_cast<int>(y_min) - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);


            //cv::imshow("output", image);
            //cv::waitKey(0);
        }
    }
};