#include <string>
#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>

#include "tokenizer.h"

int main() {
    lh::FullTokenizer tokenizer(
            "/Users/quangbd/IdeaProjects/bert-nlu-training/data/models/tinybert/vocab.txt");

    const char *text = "add sabrina salerno to the grime instrumentals playlist";
    std::vector<std::vector<std::string>> a_tokens(1);
    tokenizer.tokenize(text, &a_tokens[0], 50);
    std::vector<std::string> tokens = a_tokens[0];
    std::cout << "Len tokens: " << tokens.size() << std::endl;
    tokens.insert(tokens.begin(), "[CLS]");
    tokens.emplace_back("[SEP]");

    uint64_t input_ids[50];
    uint64_t input_mask[50] = {0};
    uint64_t segment_ids[50] = {0};
    for (int i = 0; i < 50; ++i) {
        std::cout << tokens[i] << ' ';
        uint64_t token_id = tokenizer.convert_token_to_id(tokens[i]);
        input_ids[i] = token_id;
        if (token_id > 0) {
            input_mask[i] = 1;
        }
    }
    std::cout << std::endl;

    torch::Tensor tensor = torch::rand({2, 3});
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/Users/quangbd/IdeaProjects"
                                  "/bert-nlu-training/data/models/snips_convert/model.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    auto options = torch::TensorOptions().dtype(torch::kLong);
    inputs.emplace_back(torch::from_blob(input_ids, {1, 50}, options));
    inputs.emplace_back(torch::from_blob(segment_ids, {1, 50}, options));
    inputs.emplace_back(torch::from_blob(input_mask, {1, 50}, options));
    at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();
    output = output.contiguous();
    std::vector<float> v_output(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    std::cout << '\n';

    return 0;
}
