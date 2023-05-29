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
// #include "pytorch_ffi.h"

struct OutputData {
    float* values;
    int length;
};

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


extern "C" __attribute__((visibility("default"))) __attribute__((used)) OutputData
image_model_inference(int index, unsigned char* data,int length, int width, int height,float* mean,float* std) {
    torch::jit::Module model = models.at(index);

    // Assuming your model takes a 1x3xHxW tensor as input
    // You should replace these dimensions with your actual model's input dimensions

//    auto options = torch::TensorOptions().dtype(torch::kFloat32);
//    auto tensor_image = torch::from_blob(data, {1, height, width, 3});
//    tensor_image = tensor_image.permute({0, 3, 1, 2});
//    torch::Tensor tensor_image = torch::from_blob(data, {length}, torch::kByte);

//    torch::Tensor tensor_image = torch::from_blob(data, {1, 3, height, width});
//    tensor_image[0][0] = tensor_image[0][0].sub_(mean[0]).div_(std[0]);
//    tensor_image[0][1] = tensor_image[0][1].sub_(mean[1]).div_(std[1]);
//    tensor_image[0][2] = tensor_image[0][2].sub_(mean[2]).div_(std[2]);
//    auto tensor_image=torch::ones({1, 3, 224, 224});
    // Load data into a torch::Tensor.
    torch::Tensor tensor_image = torch::from_blob(data, {height, width, 3}, torch::kByte);

    // Convert the tensor to float.
    tensor_image = tensor_image.to(torch::kFloat32);
//
//    // Normalize the data to the range [0, 1] by dividing by 255.
//    tensor_image /= 255.0;
//
    // Add batch dimension.
    tensor_image = tensor_image.unsqueeze(0);
    
    // Normalize each color channel.
//    for (int i = 0; i < 3; ++i) {
//
//        tensor_image[0][i] = (tensor_image[0][i] - mean[i]) / std[i];
//    }
    
//    tensor_image.permute({0, 3, 1, 2});
//    auto input_tensor =tensor_image.data_ptr<float>();
    auto output_tensor = model.forward({tensor_image}).toTensor();

    std::cout << output_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    // printing_buffer << "forward_output.tagKind(): " << forward_output.tagKind()  << std::endl;

    // Note that here I'm only allowing the output to be tensor, but this can be easily changed
    // Also for simplicity I only assume that output tensor has only one dimension
    
    auto results = output_tensor.sort(-1, true);
    auto softmaxs = std::get<0>(results)[0].softmax(0);
    auto indexs = std::get<1>(results)[0];

    // int tensor_length = output_tensor.sizes()[0];
    for (int i = 0; i < 20; ++i) {
      std::cout << "    ============= Top-" << i + 1
                << " =============" << std::endl;
      std::cout << "    With Probability:  "
                << softmaxs[i].item<float>() << "%" << std::endl;
    }
    // Copy the output tensor data to a new array
    int tensor_length = output_tensor.numel();

    float *output_data = static_cast<float*>(malloc(sizeof(float) * tensor_length));
    memcpy(output_data, output_tensor.data_ptr<float>(), sizeof(float) * tensor_length);

    struct OutputData output;
    output.values = output_data;
    output.length = tensor_length;

    return output;
}
