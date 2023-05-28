
// #include <torch/script.h>
// #include <string>
// #include <vector>
// #include <iostream>


int load_ml_model(const char* model_path);
char* getPrintingBufferAndClear();
float** modelInference(float* input_data_ptr);
float* image_model_inference(int index,unsigned char* data, int width, int height);