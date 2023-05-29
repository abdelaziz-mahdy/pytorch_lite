
// #include <torch/script.h>
// #include <string>
// #include <vector>
// #include <iostream>
// #include <cstdint>

struct OutputData {
    float* values;
    int length;
};


int load_ml_model(const char* model_path);
char* getPrintingBufferAndClear();
float** modelInference(float* input_data_ptr);
struct OutputData image_model_inference(int index,unsigned char* data,int length, int width, int height,float* mean,float* std);
