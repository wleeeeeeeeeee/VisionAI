#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <memory>
#include "../Log/trtLogger.h"


class fileTransformer {
private:
public:
	/**
	* @todo file conversion from onnx to tensorRT files.
	* @detail Function to build and serialize a TensorRT engine
    * @param string? getting file from string
	* @param byte reading file from binary file etc.
	*/
    static bool onnx2trt(const std::string& onnxFilePath, const std::string& engineFilePath) {
        TensorRTLogger logger;

        // Create builder, network, and config
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            std::cerr << "Failed to create TensorRT builder." << std::endl;
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create TensorRT network." << std::endl;
            return false;
        }

        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, logger));
        if (!parser) {
            std::cerr << "Failed to create ONNX parser." << std::endl;
            return false;
        }

        // Parse ONNX model
        if (!parser->parseFromFile(onnxFilePath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX file: " << onnxFilePath << std::endl;
            return false;
        }

        // Create builder config
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create TensorRT builder config." << std::endl;
            return false;
        }

        // Enable FP16 precision if supported
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        // Build serialized engine
        auto serializedModel = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (!serializedModel) {
            std::cerr << "Failed to build serialized TensorRT engine." << std::endl;
            return false;
        }

        // Write serialized model to file
        std::ofstream engineFile(engineFilePath, std::ios::binary);
        if (!engineFile) {
            std::cerr << "Failed to open engine file for writing: " << engineFilePath << std::endl;
            return false;
        }
        engineFile.write(reinterpret_cast<const char*>(serializedModel->data()),
            serializedModel->size());
        std::cout << "Engine built and saved to: " << engineFilePath << std::endl;
        return true;
    }
};