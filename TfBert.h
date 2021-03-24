//
// Created by quangbd on 23/03/2021.
//

#ifndef CBERT_NLU_TFBERT_H
#define CBERT_NLU_TFBERT_H


#include <vector>
#include "Result.h"

extern "C" {
#include "tensorflow-lite/c/c_api.h"
}

class TfBert {
private:
    // Variable
    TfLiteInterpreter *interpreter;
    TfLiteModel *model;
    TfLiteInterpreterOptions *options;
    Result result;

    // Function
    explicit TfBert(const char *model_path, const char *intent_labels_path, const char *slot_labels_path);

public:
    static TfBert &get_instance(const char *model_path, const char *intent_labels_path, const char *slot_labels_path);

    std::vector<float> predict(const uint64_t *input_ids, const uint64_t *segment_ids,
                               const uint64_t *input_mask);
};


#endif //CBERT_NLU_TFBERT_H
