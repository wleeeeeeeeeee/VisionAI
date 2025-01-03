#include "ITask.h"

class ISegmentation : public ITask {
protected:
    int cls_idx = 0;    //background class is chosen for default
    struct OutputSeg
    {
        int id;
        float score;
        cv::Rect box;
        cv::Mat mask;

    };
    struct MaskParams
    {
        int seg_channels = 32;
        int seg_width = 160;
        int seg_height = 160;
        int net_width = 640;
        int net_height = 640;
        float mask_threshold = 0.5;
        cv::Size input_shape;
        cv::Vec4d params;
    };
    static void GetMask(const cv::Mat& mask_proposals, const cv::Mat& mask_protos, OutputSeg& output, const MaskParams& mask_params)
    {
        int seg_channels = mask_params.seg_channels;
        int net_width = mask_params.net_width;
        int seg_width = mask_params.seg_width;
        int net_height = mask_params.net_height;
        int seg_height = mask_params.seg_height;
        float mask_threshold = mask_params.mask_threshold;
        cv::Vec4f params = mask_params.params;
        cv::Rect temp_rect = output.box;

        //crop from mask_protos
        int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
        int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
        int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
        int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

        rang_w = MAX(rang_w, 1);
        rang_h = MAX(rang_h, 1);
        if (rang_x + rang_w > seg_width)
        {
            if (seg_width - rang_x > 0)
                rang_w = seg_width - rang_x;
            else
                rang_x -= 1;
        }
        if (rang_y + rang_h > seg_height)
        {
            if (seg_height - rang_y > 0)
                rang_h = seg_height - rang_y;
            else
                rang_y -= 1;
        }

        std::vector<cv::Range> roi_rangs;
        roi_rangs.push_back(cv::Range(0, 1));
        roi_rangs.push_back(cv::Range::all());
        roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
        roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

        //crop
        cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
        cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
        cv::Mat matmul_res = (mask_proposals * protos).t();
        cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
        cv::Mat dest, mask;

        dest = masks_feature;

        int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
        int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
        int width = ceil(net_width / seg_width * rang_w / params[0]);
        int height = ceil(net_height / seg_height * rang_h / params[1]);

        cv::resize(dest, mask, cv::Size(width, height), cv::INTER_LINEAR);
        mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
        output.mask = mask;

        //cv::imshow("mask", output.mask);
        //cv::waitKey(0);
    }

    static void draw_result(cv::Mat& img, std::vector<OutputSeg> output_seg, cv::Mat& result_img)
    {
        cv::Mat mask = img.clone();
        result_img = img.clone();
        srand(time(0));

        for (int i = 0; i < 20; i++)
        {
            cv::rectangle(result_img, output_seg[i].box, cv::Scalar(255, 0, 0), 1);
            mask(output_seg[i].box).setTo(cv::Scalar(rand() % 256, rand() % 256, rand() % 256), output_seg[i].mask);
            std::string label = "class" + std::to_string(output_seg[i].id) + ":" + cv::format("%.2f", output_seg[i].score);
            cv::putText(result_img, label, cv::Point(output_seg[i].box.x, output_seg[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        }

        addWeighted(result_img, 0.5, mask, 0.5, 0, result_img);
    }

public:
    using InputType = cv::Mat;	// opencv Image
    using OutputType = std::vector<OutputSeg>;

    static void draw_results(cv::Mat& img, std::vector<OutputSeg> output_seg, cv::Mat& result_img) {
        draw_result(img, output_seg, result_img);
    }

    /// <summary>
    /// Image preprocessing function. Takes opencv mat image as input and resize,normalize,convert to produce matching NCHW tensor.
    /// Will be deprecated later and moved to parent class. Common functions will be inherited from parent ITask interface class.
    /// </summary>
    /// <param name="before">cv::Mat src</param>
    /// <param name="after">at::Tensor dst</param>

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
    //static void preProcess(const InputType& before, at::Tensor& after) {
    //    try {

    //        //LetterBox
    //        cv::Mat letterbox;
    //        cv::Mat temp = before.clone();
    //        cv::Vec4d m_params;
    //        LetterBox(temp, letterbox, m_params, cv::Size(640, 640));

    //        cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);

    //        torch::Tensor input;
    //        letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);
    //        input = torch::from_blob(letterbox.data, { 1, letterbox.rows, letterbox.cols, letterbox.channels() }, torch::kFloat).to(torch::kCUDA);
    //        input = input.permute({ 0, 3, 1, 2 }).contiguous();
    //        after = input.clone();
    //    }
    //    catch (std::exception& ex) {
    //        std::cout << ex.what() << std::endl;
    //    }
    //}

    

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
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<int> class_ids;
            std::vector<std::vector<float>> picked_proposals;

            for (int i = 0; i < 8400; i++) {
                float* ptr = data0_ptr + i * 116;
                float* classes_scores = ptr + 4;
                int class_id = std::max_element(classes_scores, classes_scores + 80) - classes_scores;
                float score = classes_scores[class_id];

                if (score < scoreThreshold) {
                    continue;
                }

                cv::Rect box;
                float x = ptr[0];
                float y = ptr[1];
                float w = ptr[2];
                float h = ptr[3];
                int left = int(x - 0.5 * w);
                int top = int(y - 0.5 * h);
                int width = int(w);
                int height = int(h);
                box = cv::Rect(left, top, width, height);

                boxes.push_back(box);
                scores.push_back(score);
                class_ids.push_back(class_id);

                std::vector<float> proto(ptr + 80 + 4, ptr + 80 + 36);
                picked_proposals.push_back(proto);

            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);
            cv::Rect holeImgRect(0, 0, 640, 640);

            std::vector<std::vector<float>> mask_proposals;
            
            for (int i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];
                OutputSeg output;
                output.id = class_ids[idx];
                output.score = scores[idx];
                output.box = boxes[idx] & holeImgRect;
                mask_proposals.push_back(picked_proposals[idx]);
                outputvec.push_back(output);
            }

            MaskParams mask_params;
            mask_params.params[0] = 1;
            mask_params.params[1] = 1;
            mask_params.params[2] = 0;
            mask_params.params[3] = 0;
            int shape[4] = { 1, 32, 160, 160 };
            cv::Mat output_mat1 = cv::Mat::zeros(4, shape, CV_32FC1);
            std::copy(data1_ptr, data1_ptr + 1 * 32 * 160 * 160, (float*)output_mat1.data);
            for (int i = 0; i < mask_proposals.size(); ++i)
            {
                GetMask(cv::Mat(mask_proposals[i]).t(), output_mat1, outputvec[i], mask_params);
            }
       }
        catch (std::exception& ex) {
            std::cout << "Error Occured During Segmentation PostProcess : " << ex.what()<< std::endl;
        }
    }
};
