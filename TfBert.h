//
// Created by quangbd on 23/03/2021.
//

#ifndef CBERT_NLU_TFBERT_H
#define CBERT_NLU_TFBERT_H


#include <vector>

extern "C" {
#include "tensorflow-lite/c/c_api.h"
}

struct BertResult {
    std::vector<float> intent_output;
    std::vector<float> slot_output;
};

class TfBert {
private:
    // Variable
    TfLiteInterpreter *interpreter;
    TfLiteModel *model;
    TfLiteInterpreterOptions *options;

    // Function
    explicit TfBert(const char *model_path);

public:
    static TfBert &get_instance(const char *model_path);

    BertResult predict(const uint64_t *input_ids, const uint64_t *segment_ids, const uint64_t *input_mask);
};


#endif //CBERT_NLU_TFBERT_H
