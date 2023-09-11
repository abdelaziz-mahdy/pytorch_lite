#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT
#endif


// #include <torch/script.h>
// #include <string>
// #include <vector>
// #include <iostream>
// #include <cstdint>

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

FFI_PLUGIN_EXPORT struct ModelLoadResult load_ml_model(const char* model_path);
FFI_PLUGIN_EXPORT struct OutputData modelInference(int index,float* input_data_ptr);
FFI_PLUGIN_EXPORT struct OutputData image_model_inference(int index, unsigned char* data,int input_length, int height, int width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data);
FFI_PLUGIN_EXPORT struct OutputData camera_model_inference(int index, unsigned char* data,int rotation,int isYUV, int model_image_height, int model_image_width,int camera_image_height,int camera_image_width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data);
