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

#include <vector>

std::stringstream printing_buffer;

std::vector<torch::jit::Module> models;

extern "C" __attribute__((visibility("default"))) __attribute__((used)) int
load_ml_model(const char* model_path) {
    torch::jit::Module model = torch::jit::load(model_path);
    models.push_back(model);
    return models.size() - 1;
}


// extern "C" __attribute__((visibility("default"))) __attribute__((used)) float **
// model_inference(float *input_data_ptr) {
//     std::vector<torch::jit::IValue> inputs;

//     auto options = torch::TensorOptions().dtype(torch::kFloat32);
//     auto in_tensor = torch::from_blob(input_data_ptr, {17}, options);
//     inputs.push_back(in_tensor);

//     auto forward_output = model.forward(inputs);

//     printing_buffer << "forward_output.tagKind(): " << forward_output.tagKind()  << std::endl;

//     // Note that here I'm only allowing the output to be tensor, but this can be easily changed
//     // Also for simplicity I only assume that output tensor has only one dimension
//     auto out_tensor = forward_output.toTensor();
//     int tensor_lenght = out_tensor.sizes()[0];

//     printing_buffer << "out_tensor.sizes(): " << out_tensor.sizes()  << std::endl;

//     delete[] temp_out_data_ptr;
//     temp_out_data_ptr = new float [tensor_lenght];
//     std::memcpy(temp_out_data_ptr, out_tensor.data_ptr<float>(), sizeof(float) *  tensor_lenght );

//     // Maybe not so pretty, but you need to somehow return the array with information about number of elements
//     float **out_data_with_length = new float *[2];
//     float *out_data_length = new float[1];
//     out_data_length[0] = tensor_lenght;
//     out_data_with_length[0] = temp_out_data_ptr;
//     out_data_with_length[1] = out_data_length;

//     return out_data_with_length;
// }

extern "C" __attribute__((visibility("default"))) __attribute__((used)) float*
image_model_inference(int index,unsigned char* data, int width, int height) {
    torch::jit::Module model = models.at(index);

    // Assuming your model takes a 1x3xHxW tensor as input
    // You should replace these dimensions with your actual model's input dimensions
    torch::Tensor tensor_image = torch::from_blob(data, {1, 3, height, width});

    // Create a Kwargs and put your tensor image in it
    torch::jit::Kwargs inputs;
    inputs["input"] = tensor_image;

    // Call the model's forward function and retrieve the output tensor
    torch::jit::Stack stack;
    model.forward(stack, inputs);
    torch::Tensor output_tensor = stack.front().toTensor();

    // Copy the output tensor data to a new array
    uint tensor_length = output_tensor.numel();
    float* output_data = (float*)malloc(sizeof(float) * tensor_length);
    memcpy(output_data, output_tensor.data_ptr<float>(), sizeof(float) * tensor_length);

    return output_data;
}
