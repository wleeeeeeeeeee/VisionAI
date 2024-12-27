#include "ITask.h"

class ISegmentation : public ITask {
private:

public:
    using InputType = std::vector<float>;	// Flattened image data
    using OutputType = std::vector<std::vector<float>>;	
    static void preProcess() {
        //condition if it need imagenet normalizing

        //normalizing to [0,1] by dividing 1/255
        //depending on input type(if opencv mat) convert input channels. bgr->rgb 
        // cvt data type to float 
    }
    static void postProcess(const torch::Tensor& tensor, OutputType& output) {
       
    }
};
