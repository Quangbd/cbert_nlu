//
// Created by quangbd on 23/03/2021.
//

#include "TfBert.h"

void softmax(float *input, size_t size) {
    int i;
    float m, sum, constant;

    m = -INFINITY;
    for (i = 0; i < size; ++i) {
        if (m < input[i]) {
            m = input[i];
        }
    }

    sum = 0.0;
    for (i = 0; i < size; ++i) {
        sum += exp(input[i] - m);
    }

    constant = m + log(sum);
    for (i = 0; i < size; ++i) {
        input[i] = exp(input[i] - constant);
    }
}

TfBert::TfBert(const char *model_path) {
    model = TfLiteModelCreateFromFile(model_path);
    options = TfLiteInterpreterOptionsCreate();

    // Create the interpreter.
    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);
    printf("Model was loaded\n");
}

TfBert &TfBert::get_instance(const char *model_path) {
    static TfBert tfBert(model_path);
    return tfBert;
}

BertResult TfBert::predict(const uint64_t *input_ids, const uint64_t *segment_ids, const uint64_t *input_mask) {
    TfLiteTensor *input_ids_tensor = TfLiteInterpreterGetInputTensor(interpreter, 1);
    TfLiteTensor *segment_ids_tensor = TfLiteInterpreterGetInputTensor(interpreter, 2);
    TfLiteTensor *input_mask_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor *intent_output_tensor =
            TfLiteInterpreterGetOutputTensor(interpreter, 0);
    const TfLiteTensor *slot_output_tensor =
            TfLiteInterpreterGetOutputTensor(interpreter, 1);
    TfLiteTensorCopyFromBuffer(input_ids_tensor, &input_ids[0],
                               input_ids_tensor->bytes);
    TfLiteTensorCopyFromBuffer(segment_ids_tensor, &segment_ids[0],
                               segment_ids_tensor->bytes);
    TfLiteTensorCopyFromBuffer(input_mask_tensor, &input_mask[0],
                               input_mask_tensor->bytes);
    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> intent_output((intent_output_tensor->bytes) / sizeof(float));
    std::vector<float> slot_output((slot_output_tensor->bytes) / sizeof(float));
    TfLiteTensorCopyToBuffer(intent_output_tensor, &intent_output[0], intent_output_tensor->bytes);
    TfLiteTensorCopyToBuffer(slot_output_tensor, &slot_output[0], slot_output_tensor->bytes);
    softmax(&intent_output[0], intent_output.size());

    struct BertResult bert_result;
    bert_result.intent_output = intent_output;
    bert_result.slot_output = slot_output;
    return bert_result;
}
