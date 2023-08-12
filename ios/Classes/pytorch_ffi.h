
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

struct ModelLoadResult load_ml_model(const char* model_path);
struct OutputData modelInference(int index,float* input_data_ptr);
struct OutputData image_model_inference(int index, unsigned char* data,int input_length, int height, int width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data);
struct OutputData camera_model_inference(int index, unsigned char* data,int rotation,int isYUV, int model_image_height, int model_image_width,int camera_image_height,int camera_image_width, int objectDetectionFlag, float* mean_values, float* std_values, float* output_data);
