#pragma once

#include <NvInfer.h>
#include <iostream>
#include <string>

class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Filter out messages lower than a specific severity
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] ";
            switch (severity) {
            case Severity::kINTERNAL_ERROR: std::cerr << "[ERROR] "; break;
            case Severity::kERROR: std::cerr << "[ERROR] "; break;
            case Severity::kWARNING: std::cerr << "[WARNING] "; break;
            case Severity::kINFO: std::cerr << "[INFO] "; break;
            default: std::cerr << "[UNKNOWN] "; break;
            }
            std::cerr << msg << std::endl;
        }
    }
};