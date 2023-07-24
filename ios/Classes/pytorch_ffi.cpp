#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/script.h>
#include <unistd.h>
#include <string>
#include <string>  
#include <iostream> 
#include <sstream>   
#include <cstddef>
#include <exception>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
// #include "pytorch_ffi.h"

/* OutputData Structure
 * values: An array that holds the output values from the model.
 * length: The length of the values array.
 * exception: If any exception occurs during inference, the description is stored here.
 */
struct OutputData {
    float* values;
    int length;
    const char* exception;
};



/* Struct: ModelLoadResult
 * Description: Holds the result of loading a ML model.
 *    - index: Index of the loaded model in the models vector.
 *    - exception: Exception message if an error occurred during loading, otherwise null.
 */
struct ModelLoadResult {
    int index;
    const char* exception;
};

/* Variable: models
 * Description: Vector to store loaded ML models.
 */
std::vector<torch::jit::Module> models;

/* Function: load_ml_model
 * Input: model_path - Path to the model file
 * Output: ModelLoadResult structure
 * Description: Loads a ML model from the specified file and adds it to the models vector.
 * Returns the index of the loaded model or -1 if an error occurred, along with an exception message.
 */
extern "C" __attribute__((visibility("default"))) __attribute__((used))
ModelLoadResult load_ml_model(const char* model_path) {
    struct ModelLoadResult result;
    try {
        // Load the model using torch::jit::load
        torch::jit::Module model = torch::jit::load(model_path);
        // Add the loaded model to the models vector
        models.push_back(model);

        // Store the index of the loaded model
        result.index = models.size() - 1;
        result.exception = "";  // Empty string indicates no exception occurred
    } catch (const std::exception& e) {
        // Set the index to -1 to indicate an error occurred
        result.index = -1;
                
        std::string exceptionMessage = e.what();
        
        // Allocate memory for the exception message
        result.exception = strdup(exceptionMessage.c_str());
    }
    return result;
}

/* Function: model_inference
 * Input: 
 *    - input_data_ptr: pointer to the input data
 *    - input_length: the number of elements in the input data
 * Output: OutputData structure
 * Description: This function runs inference on the given input data and returns output data.
 * It also captures any exception that might occur during inference and records it in OutputData.
 */
extern "C" __attribute__((visibility("default"))) __attribute__((used)) OutputData
model_inference(int index,float *input_data_ptr,int input_length) {
    // Define the output data structure
    struct OutputData output;
    try {
        // Load the PyTorch model
        torch::jit::Module model = models.at(index);
        // Convert input data into PyTorch tensor
        auto in_tensor = torch::from_blob(input_data_ptr, {input_length}, torch::kFloat32);
        // Run the model with the input tensor and get output tensor
        auto output_tensor = model.forward({in_tensor}).toTensor();

        // Get the number of elements in the output tensor
        int tensor_length = output_tensor.numel();

        // Allocate memory for output data and copy data from the output tensor
        float *output_data = static_cast<float*>(malloc(sizeof(float) * tensor_length));
        memcpy(output_data, output_tensor.data_ptr<float>(), sizeof(float) * tensor_length);

        // Store the output data and length in the output data structure
        output.values = output_data;
        output.length = tensor_length;
        output.exception = "";  // Empty string indicates no exception occurred
    }
    catch (const std::exception& e) {
        
        // If any exception occurs, record it in the output data structure  
        std::string exceptionMessage = e.what();
        
        // Allocate memory for the exception message
        output.exception = strdup(exceptionMessage.c_str());
    }
    // Return the output data structure
    return output;
}

/* Function: image_model_inference
 * Input: 
 *    - index: index of the model to be used for inference
 *    - data: pointer to the image data
 *    - height: the height of the image
 *    - width: the width of the image
 * Output: OutputData structure
 * Description: This function runs inference on the given image data and returns output data.
 * It also captures any exception that might occur during inference and records it in OutputData.
 */
extern "C" __attribute__((visibility("default"))) __attribute__((used)) OutputData
image_model_inference(int index, unsigned char* data,int input_length, int height, int width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data) {
    // Define the output data structure
    struct OutputData output;
    try {
        cv::_InputArray inputArray(data,input_length); // No size is given here

        cv::Mat img = cv::imdecode(inputArray, cv::IMREAD_COLOR);
        
        cv::Mat imgRGB;
        cv::cvtColor(img, imgRGB, cv::COLOR_BGR2RGB);
        // Convert the received image data into an OpenCV Mat
//        cv::Mat img(height, width, CV_8UC3, data);

        // Use the provided height and width as the desired size
        cv::Size sizeDesired(width, height);

        // Resize the image to the desired size
        cv::Mat imgResized;
        cv::resize(imgRGB, imgResized, sizeDesired);

//        // Convert image to float and normalize to [0, 1]
        cv::Mat imgFloat;
        // convert [unsigned int] to [float]
        imgResized.convertTo(imgFloat, CV_32FC3, 1.0f / 255.0f);
//        // Initialize cv::Mat for mean and std values
//        cv::Mat mean = (cv::Mat_<float>(1, 3) << mean_values[0], mean_values[1], mean_values[2]);
//        cv::Mat std = (cv::Mat_<float>(1, 3) << std_values[0], std_values[1], std_values[2]);
//
//        cv::Mat mean_f, std_f;
//
//        mean.convertTo(mean_f, CV_64F);
//        std.convertTo(std_f, CV_64F);
//
//        // Normalize the resized image with provided mean and std values
//        cv::Mat imgNormalized;
//        cv::subtract(imgFloat, mean_f, imgNormalized);
//        cv::divide(imgNormalized, std_f, imgNormalized);
//
////        cv::subtract(imgFloat, mean, imgNormalized);
//        cv::divide(imgNormalized, std, imgNormalized);

        // Load the PyTorch model
        torch::jit::Module model = models.at(index);

        // Convert the normalized image data into a PyTorch tensor
        torch::Tensor tensor_image = torch::from_blob(imgFloat.data, {1, 3, height, width}, torch::kFloat32);
        // Channel-wise mean and std values
        torch::Tensor mean = torch::tensor({mean_values[0], mean_values[1], mean_values[2]});
        torch::Tensor std = torch::tensor({std_values[0], std_values[1], std_values[2]});

        // Convert mean and std to the same device and dtype as tensor_image
        mean = mean.to(tensor_image.device()).to(tensor_image.dtype());
        std = std.to(tensor_image.device()).to(tensor_image.dtype());

        // Expand dimensions of mean and std to match with the tensor_image shape for broadcasting
        mean = mean.view({1, 3, 1, 1});
        std = std.view({1, 3, 1, 1});

        // Normalize
        tensor_image = tensor_image.sub(mean).div(std);
        torch::Tensor output_tensor;
        if (objectDetectionFlag==1) {
            // Run the object detection model

//            // Run the model with the input tensor and get the output tuple of tensors
//            auto output_tuple_ptr = model.forward({tensor_image}).toTuple();
//
//            // Example: Extract the first tensor from the tuple
//            auto& output_tuple = *output_tuple_ptr;
//
//            // Access the elements of the tuple using the appropriate methods
//            output_tensor = output_tuple.elements()[0].toTensor();

            torch::jit::IValue outputTuple = model.forward({tensor_image}).toTuple();
            output_tensor = outputTuple.toTuple()->elements()[0].toTensor();

        }
        else {
            // Run the model with the input tensor and get the output tensor
            output_tensor = model.forward({tensor_image}).toTensor();

        }
        // Get the number of elements in the output tensor
        int tensor_length = output_tensor.numel();
        
        // Allocate memory for output data and copy data from the output tensor
        // float *output_data = static_cast<float*>(malloc(sizeof(float) * tensor_length));
        memcpy(output_data, output_tensor.data_ptr<float>(), sizeof(float) * tensor_length);

        // Store the output data and length in the output data structure
        output.values = output_data;
        output.length = tensor_length;
        output.exception = "";  // Empty string indicates no exception occurred
    }
    catch (const std::exception& e) {
        // If any exception occurs, record it in the output data structure  
        std::string exceptionMessage = e.what();
        
        // Allocate memory for the exception message
        output.exception = strdup(exceptionMessage.c_str());
    }
    // Return the output data structure
    return output;
}

