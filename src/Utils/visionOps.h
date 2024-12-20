#pragma once
#include <torch/torch.h>


class visionOps {
private:
public:
    // Function to calculate IoU
    static torch::Tensor calculate_iou(const torch::Tensor& boxes1, const torch::Tensor& boxes2) {
        std::cout << boxes1.size(0) << std::endl;
        std::cout << boxes1.size(1) << std::endl;
        std::cout << boxes2.size(0) << std::endl;
        std::cout << boxes2.size(1) << std::endl;
        auto x1 = torch::max(boxes1.index({ torch::indexing::Slice(), 0 }), boxes2.index({ torch::indexing::Slice(), 0 }));
        auto y1 = torch::max(boxes1.index({ torch::indexing::Slice(), 1 }), boxes2.index({ torch::indexing::Slice(), 1 }));
        auto x2 = torch::min(boxes1.index({ torch::indexing::Slice(), 2 }), boxes2.index({ torch::indexing::Slice(), 2 }));
        auto y2 = torch::min(boxes1.index({ torch::indexing::Slice(), 3 }), boxes2.index({ torch::indexing::Slice(), 3 }));

        auto intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0);
        auto area1 = (boxes1.index({ torch::indexing::Slice(), 2 }) - boxes1.index({ torch::indexing::Slice(), 0 })) *
            (boxes1.index({ torch::indexing::Slice(), 3 }) - boxes1.index({ torch::indexing::Slice(), 1 }));
        auto area2 = (boxes2.index({ torch::indexing::Slice(), 2 }) - boxes2.index({ torch::indexing::Slice(), 0 })) *
            (boxes2.index({ torch::indexing::Slice(), 3 }) - boxes2.index({ torch::indexing::Slice(), 1 }));

        return intersection / (area1 + area2 - intersection);
    }
    // NMS function
    static torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores, double iou_threshold) {
        // Sort scores in descending order
        auto sorted_indices = std::get<1>(scores.sort(0, /*descending=*/true));

        std::vector<int64_t> keep_indices;
        auto selected_boxes = boxes.index({ sorted_indices });
        auto selected_scores = scores.index({ sorted_indices });

        std::cout << boxes.size(0) << std::endl;
        std::cout << boxes.size(1) << std::endl;

        while (sorted_indices.size(0) > 0) {
            // Select the top score index
            auto current_index = sorted_indices[0].item<int64_t>();
            keep_indices.push_back(current_index);

            if (sorted_indices.size(0) == 1) break;


            std::cout << boxes.size(0) << std::endl;
            std::cout << boxes.size(1) << std::endl;
            // Compare the top box with the rest
            auto current_box = boxes[current_index].unsqueeze(0);
            auto remaining_boxes = boxes.index({ sorted_indices.slice(0, 1) });
            auto iou = calculate_iou(current_box, remaining_boxes);

            // Filter out boxes with IoU above the threshold
            auto mask = iou <= iou_threshold;
            auto indices = torch::nonzero(mask).squeeze(1);
            sorted_indices = sorted_indices.index({indices});
        }

        return torch::tensor(keep_indices, torch::dtype(torch::kInt64));
    }
};