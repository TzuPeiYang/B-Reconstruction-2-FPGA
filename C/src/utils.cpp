#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "utils.h"


void read_data_config(const char *filename) {
    try {
        // Load the YAML file
        YAML::Node config = YAML::LoadFile(filename);

        // --- new_variables (if present) ---
        if (config["new_variables"]) {
            std::cout << "New Variables Section Found\n";
        }

        // --- preprocess section ---
        if (config["preprocess"]) {
            auto preprocess = config["preprocess"];
            std::cout << "Preprocess method: " << preprocess["method"].as<std::string>() << "\n";
            std::cout << "Data fraction: " << preprocess["data_fraction"].as<double>() << "\n";
        }

        // --- inputs section ---
        if (config["inputs"]) {
            auto inputs = config["inputs"];
            for (auto it : inputs) {
                std::string input_name = it.first.as<std::string>();
                auto input_node = it.second;
                std::cout << "\nInput block: " << input_name << "\n";

                if (input_node["length"]) {
                    std::cout << "  length = " << input_node["length"].as<int>() << "\n";
                }
                if (input_node["pad_mode"]) {
                    std::cout << "  pad_mode = " << input_node["pad_mode"].as<std::string>() << "\n";
                }
                if (input_node["pad_value"]) {
                    std::cout << "  pad_value = " << input_node["pad_value"].as<int>() << "\n";
                }

                if (input_node["vars"]) {
                    std::cout << "  vars:\n";
                    for (auto var : input_node["vars"]) {
                        if (var.IsScalar()) {
                            // e.g. Part_E
                            std::cout << "    - " << var.as<std::string>() << "\n";
                        } else if (var.IsSequence()) {
                            std::cout << "    - [ ";
                            for (auto v : var) {
                                if (v.IsNull()) std::cout << "null ";
                                else std::cout << v.as<std::string>() << " ";
                            }
                            std::cout << "]\n";
                        }
                    }
                }
            }
        }

        // --- labels section ---
        if (config["labels"]) {
            auto labels = config["labels"];
            std::cout << "\nLabels type: " 
                    << labels["type"].as<std::string>() << "\n";

            std::cout << "Labels values:\n";
            for (auto val : labels["value"]) {
                std::cout << "  - " << val.as<std::string>() << "\n";
            }
        }

        // --- observers section ---
        if (config["observers"]) {
            std::cout << "\nObservers:\n";
            for (auto obs : config["observers"]) {
                std::cout << "  - " << obs.as<std::string>() << "\n";
            }
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error: " << e.what() << "\n";
    }

}