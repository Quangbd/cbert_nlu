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

std::string Result::convert(std::vector<std::string> tokens, const BertResult &bert_result) {
    nlohmann::json result_json;

    // for intent
    std::vector<float> intent_output = bert_result.intent_output;
    auto intent_max_score = std::max_element(intent_output.begin(), intent_output.end());
    auto intent_max_index = intent_max_score - intent_output.begin();
    result_json["intent"]["intent_name"] = intent_labels[intent_max_index];
    result_json["intent"]["probability"] = roundf(*intent_max_score);

    // for slot
    std::vector<float> slot_output = bert_result.slot_output;
    auto slot_size = slot_labels.size();
    std::vector<std::string> values;
    std::vector<std::string> slot_names;
    int count = 0;
    for (int i = 1; i < tokens.size() - 1; ++i) {
        auto slot_max_index = std::max_element(
                slot_output.begin() + i * slot_size, slot_output.begin() + (i + 1) * slot_size) -
                              (slot_output.begin() + i * slot_size);
        std::string value = tokens[i];
        std::string slot_name = slot_labels[slot_max_index];

        if ((count > 0) && (value[0] == '#') && (value[1] == '#')) {
            values[count - 1] = values[count - 1].append(value.replace(0, 2, ""));
            printf("%s\n", value.c_str());
        } else {
            values.push_back(value);
            slot_names.push_back(slot_name);
            count++;
        }
    }

    result_json["tokens"] = values;
    result_json["slot_labels"] = slot_names;

    return result_json.dump();
}

Result::Result() = default;


