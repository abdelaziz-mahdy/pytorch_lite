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
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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
cv::Mat decode_image(unsigned char* data, int input_length) {
    cv::_InputArray inputArray(data, input_length);
    return cv::imdecode(inputArray, cv::IMREAD_COLOR);
}
cv::Mat preprocess_mat(cv::Mat imgResized, float* mean_values, float* std_values) {


    imgResized.convertTo(imgResized, CV_32FC3, 1.0f / 255.0f);

    cv::Mat mean(1, 1, CV_32FC3, cv::Scalar(mean_values[0], mean_values[1], mean_values[2]));
    cv::Mat std(1, 1, CV_32FC3, cv::Scalar(std_values[0], std_values[1], std_values[2]));

    std::vector<cv::Mat> channels;
    cv::split(imgResized, channels);

    for (int i = 0; i < 3; ++i) {
        cv::Mat mean_ch = mean.at<cv::Vec3f>(0)[i] * cv::Mat::ones(imgResized.size(), CV_32F);
        cv::Mat std_ch = std.at<cv::Vec3f>(0)[i] * cv::Mat::ones(imgResized.size(), CV_32F);

        cv::subtract(channels[i], mean_ch, channels[i]);
        cv::divide(channels[i], std_ch, channels[i]); 
    }

    cv::Mat imgNormalized;
    cv::merge(channels, imgNormalized);
    return imgNormalized;
}
OutputData run_inference(int index, cv::Mat imgNormalized, int height, int width, int objectDetectionFlag, float* output_data) {
    OutputData output;
    try {
        torch::jit::Module model = models.at(index);
        torch::Tensor tensor_image = torch::from_blob(imgNormalized.data, {1, height, width, 3}, torch::kFloat32);
        tensor_image = tensor_image.permute({0, 3, 1, 2});

        torch::Tensor output_tensor;
        if (objectDetectionFlag == 1) {
            torch::jit::IValue outputTuple = model.forward({tensor_image}).toTuple();
            output_tensor = outputTuple.toTuple()->elements()[0].toTensor();
        } else {
            output_tensor = model.forward({tensor_image}).toTensor();
        }

        int tensor_length = output_tensor.numel();
        memcpy(output_data, output_tensor.data_ptr<float>(), sizeof(float) * tensor_length);

        output.values = output_data;
        output.length = tensor_length;
        output.exception = "";

    } catch (const std::exception& e) {
        std::string exceptionMessage = e.what();
        output.exception = strdup(exceptionMessage.c_str());
    }
    return output;
}
void rotateMat(cv::Mat &matImage, int rotation) {
    if (rotation == 90) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 1); //transpose+flip(1)=CW
    } else if (rotation == 270) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 0); //transpose+flip(0)=CCW
    } else if (rotation == 180) {
        flip(matImage, matImage, -1);    //flip(-1)=180
    }
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

 image_model_inference(int index, unsigned char* data, int input_length, int height, int width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data) {
    cv::Mat img = decode_image(data, input_length);
    cv::Mat imgRGB;
    cv::cvtColor(img, imgRGB, cv::COLOR_BGR2RGB);

    cv::Size sizeDesired(width, height);
    cv::Mat imgResized;
    cv::resize(imgRGB, imgResized, sizeDesired);
    cv::Mat imgNormalized = preprocess_mat(imgResized, mean_values, std_values);
    return run_inference(index, imgNormalized, height, width, objectDetectionFlag, output_data);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used)) OutputData

 camera_model_inference(int index, unsigned char* data,int rotation,int isYUV, int model_image_height, int model_image_width,int camera_image_height,int camera_image_width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data) {
    cv::Mat img;
    if (isYUV==1) {
        cv::Mat yuvMat(camera_image_height + camera_image_height / 2, camera_image_width, CV_8UC1, data);
        cv::cvtColor(yuvMat, img, cv::COLOR_YUV2BGRA_NV21);
    } else {
        img = cv::Mat(camera_image_height, camera_image_width, CV_8UC4, data);
    }
    cv::Mat imgRGB;
    cv::cvtColor(img, imgRGB, cv::COLOR_BGRA2RGB);

    cv::Size sizeDesired(model_image_width, model_image_height);
    cv::Mat imgResized;
    cv::resize(imgRGB, imgResized, sizeDesired);


    rotateMat(imgResized, rotation);

    cv::Mat imgNormalized = preprocess_mat(imgResized, mean_values, std_values);
    return run_inference(index, imgNormalized, model_image_height, model_image_width, objectDetectionFlag, output_data);
}
