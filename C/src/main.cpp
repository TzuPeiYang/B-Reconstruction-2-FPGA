#include <iostream>
#include <network.h>
#include <utils.h>


int main(int argc, char *argv[]){
    const char *config_file = argv[1];
    std::vector<std::string> root_files;
    root_files.push_back("/home/tpyang/B-Reconstruction-2-FPGA/python/gen_level_2B/data/test.root");

    load_root_files(config_file, root_files, "pf_points");
    load_root_files(config_file, root_files, "pf_features");

    return 0;
}