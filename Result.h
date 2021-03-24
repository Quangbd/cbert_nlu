//
// Created by quangbd on 24/03/2021.
//

#ifndef CBERT_NLU_RESULT_H
#define CBERT_NLU_RESULT_H

#include <vector>
#include <string>

class Result {
private:
    std::vector<std::string> intent_labels;
    std::vector<std::string> slot_labels;

public:
    Result(const char *intent_labels_path, const char *slot_labels_path);

    Result();

    std::string convert(std::vector<float> intent_output, std::vector<float> slot_output);
};


#endif //CBERT_NLU_RESULT_H
