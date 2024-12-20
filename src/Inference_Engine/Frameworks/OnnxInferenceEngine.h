#pragma once
#include "../IInferenceEngine/IInferenceEngine.h"
#include <cuda_runtime_api.h>
#include <cpu_provider_factory.h>
#include "../../Utils/dataTransformation.h"

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
	std::vector<Ort::AllocatedStringPtr> input_node_names_ptr;
	std::vector<const char*> output_node_names;
	std::vector<Ort::AllocatedStringPtr> output_node_names_ptr;
	std::vector<std::vector<int64_t>> input_dims;
	std::vector<std::vector<int64_t>> output_dims;

	
	void checkCudaDevices() {
		int device_count = 0;
		cudaError_t err = cudaGetDeviceCount(&device_count);
		if (err != cudaSuccess) {
			std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			return;
		}

		std::cout << "Number of CUDA devices: " << device_count << std::endl;
		for (int i = 0; i < device_count; ++i) {
			cudaDeviceProp device_prop;
			err = cudaGetDeviceProperties(&device_prop, i);
			if (err != cudaSuccess) {
				std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
				continue;
			}
			std::cout << "Device " << i << ": " << device_prop.name << std::endl;
		}
	}

	void get_model(const std::string& model_path, std::string& device, int threads = 4, int gpu_mem_limit = 4) {
		try {
			auto availableProviders = Ort::GetAvailableProviders();
			for (const auto& provider : availableProviders) {
				std::cout << provider << " ";
			}
			std::cout << std::endl;

			checkCudaDevices();

			

			Ort::SessionOptions sessionOptions;
			sessionOptions.SetIntraOpNumThreads(threads);
			sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			//cuda
			//OrtCUDAProviderOptions cuda_options;
			//cuda_options.device_id = 0;
			//cuda_options.arena_extend_strategy = 0;
			//cuda_options.gpu_mem_limit = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // gpu memory limit
			//cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
			//cuda_options.do_copy_in_default_stream = 1;
			//sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
			
			OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
			//OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, 1);
			//sessionOptions.AppendExecutionProvider(1);

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
		this->input_nums = this->session.GetInputCount();
		this->output_nums = this->session.GetOutputCount();

		for (int i = 0; i < this->input_nums; i++) {
			Ort::AllocatedStringPtr input_name = this->session.GetInputNameAllocated(i, this->allocator);
			this->input_node_names.push_back(input_name.get());
			this->input_node_names_ptr.push_back(std::move(input_name));

			auto input_shape_info = this->session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
			this->input_dims.push_back(input_shape_info.GetShape());
		}
		for (int i = 0; i < this->output_nums; i++) {
			Ort::AllocatedStringPtr output_name = this->session.GetOutputNameAllocated(i, this->allocator);
			this->output_node_names.push_back(output_name.get());
			this->output_node_names_ptr.push_back(std::move(output_name));

			auto output_shape_info = this->session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
			this->output_dims.push_back(output_shape_info.GetShape());
		}

		//printing input shape
		bool has_negative = false;
		for (int i = 0; i < this->input_nums; ++i) {
			std::cout << "input_dims: ";
			for (auto& j : this->input_dims[i]) {
				if (j < 0) has_negative = true; // dynamic batch(if the batch size is -1)
				std::cout << j << " ";
			}
			std::cout << std::endl;
		}

		//printing outputput shape
		for (int i = 0; i < this->output_nums; ++i) {
			std::cout << "output_dims: ";
			for (const auto j : this->output_dims[i]) {
				std::cout << j << " ";
				if (j < 0) has_negative = true;
			}
			std::cout << std::endl;
		}


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
			cv::Mat input_tensor;
			TaskType::preProcess(input, input_tensor);

			//converting to ort::value 
			cv::Mat blob = cv::dnn::blobFromImage(input_tensor);

			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
			Ort::Value ort_input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims[0].data(), input_dims[0].size());
			
			std::vector<Ort::Value> output_tensors;

			//run inference
			output_tensors = this->session.Run(
				this->runOptions,
				this->input_node_names.data(),
				&ort_input_tensor,
				this->input_nums,
				this->output_node_names.data(),
				this->output_nums
			);
			
			float* output_data = output_tensors[0].GetTensorMutableData<float>();
			std::vector<float> output_vec(output_data, output_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());

			at::Tensor output_tensor;
			dataTransformer::vec2tensor(output_vec, output_dims[0], output_tensor);

			// Assuming 'tensor' is your tensor object
			std::vector<int64_t> tensor_shape = output_tensor.sizes().vec();

			// Print the shape
			std::cout << "Tensor shape: [";
			for (size_t i = 0; i < tensor_shape.size(); ++i) {
				std::cout << tensor_shape[i];
				if (i < tensor_shape.size() - 1) {
					std::cout << ", ";
				}
			}
			std::cout << "]" << std::endl;


			TaskType::postProcess(output_tensor, output);

		}
		catch (std::exception& ex) {
			// somekind of exception error printed on log
			std::cout << ex.what() << std::endl;
		}
	}
};