#include "ITask.h"

class IClassification : public ITask {

private:
protected:
    //normalization
    //resizing
    //cropping
    //Binarization(if needed)

    //confidence thresholding
    //soft max activation
    //top-k predictions
public:
    using InputType = std::vector<float>;	// Flattened image data
    using OutputType = std::vector<std::vector<float>>;
    static void preProcess() {

    }
    static void postProcess(const torch::Tensor& tensor, OutputType& output) {

    }
};