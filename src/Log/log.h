#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>


class LoggerManager_basic {
public:
    // Get the singleton instance
    static std::shared_ptr<spdlog::logger> getLogger() {
        static auto logger = createLogger();
        return logger;
    }

    // Set log level dynamically
    static void setLogLevel(spdlog::level::level_enum level) {
        getLogger()->set_level(level);
    }
private:
    // Create a logger with console and file sinks
    static std::shared_ptr<spdlog::logger> createLogger() {
        // Console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info); // Default console log level

        // File sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("sdk_log.txt", true);
        file_sink->set_level(spdlog::level::debug); // Default file log level

        // Combine sinks into one logger
        std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
        auto logger = std::make_shared<spdlog::logger>("VisionAI", sinks.begin(), sinks.end());

        spdlog::register_logger(logger);
        return logger;
    }
};

class LoggerManager_async {
public:
    static std::shared_ptr<spdlog::logger> getLogger() {
        static auto logger = createAsyncLogger();
        return logger;
    }

private:
    static std::shared_ptr<spdlog::logger> createAsyncLogger() {
        spdlog::init_thread_pool(8192, 1); // Queue size: 8192, 1 worker thread

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("sdk_log.txt", true);

        std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
        auto logger = std::make_shared<spdlog::async_logger>(
            "VisionAI", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);

        spdlog::register_logger(logger);
        return logger;
    }
};
