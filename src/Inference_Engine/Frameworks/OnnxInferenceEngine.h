#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"

template <typename TaskType>
class OnnxInferenceEngine : public IInferenceEngine<TaskType> {
private:
	Ort::Env env{};
	Ort::AllocatorWithDefaultOptions allocator{};
	Ort::RunOptions runOptions{};
	Ort::Session session = Ort::Session(nullptr);
	size_t input_nums{};
	size_t output_nums{};
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;

	void get_model(const std::string& model_path, std::string& device, int threads = 0, int gpu_mem_limit = 4) {
		try {
			auto availableProviders = Ort::GetAvailableProviders();
			for (const auto& provider : availableProviders) {
				std::cout << provider << " ";
			}
			std::cout << std::endl;

			Ort::SessionOptions sessionOptions;
			sessionOptions.SetIntraOpNumThreads(threads);
			sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			//cuda
			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = 0;
			cuda_options.arena_extend_strategy = 0;
			cuda_options.gpu_mem_limit = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // gpu memory limit
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
			cuda_options.do_copy_in_default_stream = 1;
			sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

			wchar_t* model_path1 = new wchar_t[model_path.size()];
			swprintf(model_path1, 4096, L"%S", model_path.c_str());

			this->session = Ort::Session(this->env, model_path1, sessionOptions);
		}
		catch (std::exception ex) {
			// somekind of exception error printed on log
			std::cout << "inside onnxInferenceengine, get_model() function was called, but {0} error occured " << ex.what() << std::endl;
		}
	}

	void get_onnx_info() {

	}

public:
	OnnxInferenceEngine() = default;
	OnnxInferenceEngine(const std::string& modelPath) {
		loadModel(modelPath);
	}
	void loadModel(const std::string& modelPath) override {
		//load model into session + configure gpu, cpu, any necessary inference options
		std::string device = "cuda";
		get_model(modelPath, device);
		
		//get onnx info (node names, input shape, output shape etc)
		get_onnx_info();

	}

	void infer(const typename TaskType::InputType& input, typename TaskType::OutputType& output) override {
		try {
			//prepare input tensor
			TaskType::preProcess();
			//run inference
			/*output_tensors = session.Run(
				this->runOptions,
				this->input_node_names.data(),
				&input_tensor,
				this->input_nums,
				this->output_node_names.data(),
				this->output_nums
			);*/
			//task-specific postprocessing
			//TaskType::processOutput(input, output);
			//TaskType::processOuput(rawOutput, output);
		}
		catch (std::exception& ex) {
			// somekind of exception error printed on log
		}
	}
};