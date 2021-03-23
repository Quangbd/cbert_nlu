//
// Created by quangbd on 23/03/2021.
//

#ifndef CBERT_NLU_TFBERT_H
#define CBERT_NLU_TFBERT_H


#include <vector>

extern "C" {
#include "includes/tensorflow-lite/c/c_api.h"
}

class TfBert {
private:
    // Variable
    TfLiteInterpreter *interpreter;
    TfLiteModel *model;
    TfLiteInterpreterOptions *options;

    // Function
    TfBert(const char *model_buffer, size_t model_size);

public:
    static TfBert &get_instance(const char *model_buffer, size_t model_size);

    std::vector<float> predict(const uint64_t *input_ids, const uint64_t *segment_ids,
                               const uint64_t *input_mask);
};


#endif //CBERT_NLU_TFBERT_H
