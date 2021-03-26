#include <string>
#include <vector>

#include "Result.h"
#include "TfBert.h"
#include "tokenizer.h"
#include <chrono>

int main() {
    lh::FullTokenizer tokenizer("../data/vocab.txt");

    const char *text = "add sabrina salerno to the grime instrumentals playlist";

    std::vector<std::vector<std::string>> a_tokens(1);
    tokenizer.tokenize(text, &a_tokens[0], 50);
    std::vector<std::string> tokens = a_tokens[0];
    printf("Len tokens: %lu\n", tokens.size());
    tokens.insert(tokens.begin(), "[CLS]");
    tokens.emplace_back("[SEP]");

    uint64_t input_ids[50] = {0};
    uint64_t input_mask[50] = {0};
    uint64_t segment_ids[50] = {0};
    printf("Input: ");
    for (int i = 0; i < tokens.size(); ++i) {
        printf("%s ", tokens[i].c_str());
        uint64_t token_id = tokenizer.convert_token_to_id(tokens[i]);
        input_ids[i] = token_id;
        if (token_id > 0) {
            input_mask[i] = 1;
        }
    }
    printf("\n");

    TfBert tfBert = TfBert::get_instance("../data/snips/model.tflite");
    BertResult nlu_result = tfBert.predict(input_ids, segment_ids, input_mask);
    long start = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < 1000; i++) {
        tfBert.predict(input_ids, segment_ids, input_mask);
    }
    long end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    Result result("../data/snips/intent_label.txt", "../data/snips/slot_label.txt");
    std::string final_result = result.convert(tokens, nlu_result);
    printf("Result:\n%s", final_result.c_str());
    std::cout << (end - start) / 1000 << "\n";

    return 0;
}
