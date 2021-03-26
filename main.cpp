#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include "wav_data.h"

int main() {
    float input_wav[80000] = WAVE_DATA;

    torch::Tensor tensor = torch::rand({2, 3});
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../model.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    inputs.emplace_back(torch::from_blob(input_wav, {1, 80000}, options));
    at::Tensor output = module.forward(inputs).toTensor();
    output = output.contiguous();
    std::vector<float> v_output(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    
    for (int i=0;i<v_output.size();i++){
        std::cout<<v_output[i];
        std::cout << '\n';
    }
    std::cout << '\n';

    return 0;
}
