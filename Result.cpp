//
// Created by quangbd on 24/03/2021.
//

#include "Result.h"
#include "json.h"
#include <fstream>

std::vector<std::string> read_label_file(const char *file_path) {
    std::vector<std::string> result;
    std::ifstream file(file_path);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
//            printf("%s\n", line.c_str());
            result.push_back(line);
        }
        file.close();
    }
    return result;
}

Result::Result(const char *intent_labels_path, const char *slot_labels_path) {
    intent_labels = read_label_file(intent_labels_path);
    slot_labels = read_label_file(slot_labels_path);
    printf("Slot and intent labels were loaded\n");
}

std::string Result::convert(std::vector<float> intent_output, std::vector<float> slot_output) {
    nlohmann::json result_json;

    // for intent
    auto intent_max_score = std::max_element(intent_output.begin(), intent_output.end());
    auto intent_max_index = intent_max_score - intent_output.begin();
    result_json["intent"]["intent_name"] = intent_labels[intent_max_index];
    result_json["intent"]["probability"] = roundf(*intent_max_score);

    // for slot


    return result_json.dump();
}

Result::Result() = default;


