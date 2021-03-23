//
// Created by quangbd on 23/03/2021.
//

#include "TfBert.h"

TfBert::TfBert(const char *model_buffer, size_t model_size) {
    model = TfLiteModelCreate(model_buffer, model_size);
    options = TfLiteInterpreterOptionsCreate();

    // Create the interpreter.
    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);
}

TfBert &TfBert::get_instance(const char *model_buffer, size_t model_size) {
    static TfBert tfBert(model_buffer, model_size);
    return tfBert;
}

std::vector<float>
TfBert::predict(const uint64_t *input_ids, const uint64_t *segment_ids, const uint64_t *input_mask) {
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
    TfLiteTensorCopyToBuffer(intent_output_tensor, &intent_output[0],
                             intent_output_tensor->bytes);
    TfLiteTensorCopyToBuffer(slot_output_tensor, &slot_output[0],
                             slot_output_tensor->bytes);
    slot_output.insert(slot_output.end(), intent_output.begin(), intent_output.end());
    return slot_output;
}