#include "ITask.h"

class IClassification : public ITask {

private:

public:
    using InputType = std::vector<float>;	// Flattened image data
    using OutputType = std::vector<std::vector<float>>;
    static void preProcess() {

    }
    static void postProcess(const torch::Tensor& tensor, OutputType& output) {

    }
};