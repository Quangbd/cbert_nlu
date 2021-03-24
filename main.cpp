#include <string>
#include <vector>

#include "Result.h"
#include "TfBert.h"
#include "tokenizer.h"

int main() {
    lh::FullTokenizer tokenizer("../data/vocab.txt");

    const char *text = "add sabrina salerno to the grime instrumentals playlist";

    std::vector<std::vector<std::string>> a_tokens(1);
    tokenizer.tokenize(text, &a_tokens[0], 50);
    std::vector<std::string> tokens = a_tokens[0];
    printf("Len tokens: %lu\n", tokens.size());
    tokens.insert(tokens.begin(), "[CLS]");
    tokens.emplace_back("[SEP]");

    uint64_t input_ids[50];
    uint64_t input_mask[50] = {0};
    uint64_t segment_ids[50] = {0};
    for (int i = 0; i < 50; ++i) {
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
    Result result("../data/snips/intent_label.txt","../data/snips/slot_label.txt");
    std::string final_result = result.convert(tokens, nlu_result);

    return 0;
}
