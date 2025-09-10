#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>
#include <TBranch.h>

#include "utils.h"

/* 
Funcntion to load root files:
inputs:
  - config_file: path to the yaml configuration file
  - root_files: vector of root file paths
  - key_name: name of the key in the yaml file
*/
void load_root_files(const char *config_file, std::vector<std::string>& root_files, const char *key) {
    std::vector<std::string> branches;
    std::vector<float> shifts;
    std::vector<float> scales;
    std::vector<float> mins;
    std::vector<float> maxs;
    std::vector<int> lengths;

    YAML::Node config = YAML::LoadFile(config_file);
    // read inputs
    if (config["inputs"]) {
        // read key
        if (!config["inputs"][key]) {
            std::cerr << "No " << key << " key found in inputs." << std::endl;
            return;
        }
        YAML::Node pf_points = config["inputs"]["pf_points"];
        if (!pf_points["length"] || !pf_points["vars"]) {
            std::cerr << "pf_points must contain length and vars." << std::endl;
            return;
        }
        lengths.push_back(pf_points["length"].as<int>());
        for (std::size_t i = 0; i < pf_points["vars"].size(); i++) {
            auto entry = pf_points["vars"][i]; 

            // [branch name, shift, scale, min, max]
            if (!entry.IsSequence()) {
                branches.push_back(entry.as<std::string>());
                shifts.push_back(0.0f);
                scales.push_back(1.0f);
                mins.push_back(-5.0f);
                maxs.push_back(5.0f);
            }
            else{
                if (entry.size() != 5) {
                    std::cerr << "Each entry in pf_points vars must have 5 elements." << std::endl;
                    return;
                }
                branches.push_back(entry[0].as<std::string>());
                if (entry[1].IsNull()) {
                    shifts.push_back(0.0f);
                } 
                else {
                    shifts.push_back(entry[1].as<float>());
                }
                if (entry[2].IsNull()) {
                    scales.push_back(1.0f);
                } 
                else {
                    scales.push_back(entry[2].as<float>());
                }
                if (entry[3].IsNull()) {
                    mins.push_back(-5.0f);
                } 
                else {
                    mins.push_back(entry[3].as<float>());
                }
                if (entry[4].IsNull()) {
                    maxs.push_back(5.0f);
                } 
                else {
                    maxs.push_back(entry[4].as<float>());
                }
            }
        }
    }
    else {
        std::cerr << "No input key found in configuration file." << std::endl;
    }
    std::cout << "length of " << key << ": " << lengths[0] << std::endl;
    for (size_t i = 0; i < branches.size(); i++) {
        std::cout << "[Branch: " << branches[i] 
                  << ", Shift: " << shifts[i] 
                  << ", Scale: " << scales[i] 
                  << ", Min: " << mins[i] 
                  << ", Max: " << maxs[i] << "]" << std::endl;
    }

    // read root files
    for (std::size_t i = 0; i < root_files.size(); i++) {
        std::cout << "Reading root file: " << root_files[i] << std::endl;
        TFile* file = TFile::Open(root_files[i].c_str(), "READ");
        if (!file || file->IsZombie()) {
            std::cerr << "Handle error: file not found or corrupted" << std::endl;
            if (file) file->Close();
            return;
        }
        TTree* tree = dynamic_cast<TTree*>(file->Get("Events"));
        if (!tree) {
            std::cerr << "Handle error: TTree not found" << std::endl; 
            file->Close();
            return;
        }
        std::vector<std::vector<double>*> branches_data(branches.size(), nullptr);
        for (size_t j = 0; j < branches.size(); j++) {
            branches_data[j] = new std::vector<double>();
            tree->SetBranchAddress(branches[j].c_str(), &branches_data[j]);
        }
    }
}