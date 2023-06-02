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
    throw std::runtime_error("An error occurred");

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
image_model_inference(int index, unsigned char* data, int height, int width) {
    // Define the output data structure
    struct OutputData output;
    try {
        // Load the PyTorch model
        torch::jit::Module model = models.at(index);

        // Convert image data into PyTorch tensor
        torch::Tensor tensor_image = torch::from_blob(data, {1,3,height, width}, torch::kFloat32);

        // Run the model with the input tensor and get the output tensor
        auto output_tensor = model.forward({tensor_image}).toTensor();

        // Get the number of elements in the output tensor
        int tensor_length = output_tensor.numel();
    throw std::runtime_error("An error occurred");

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
