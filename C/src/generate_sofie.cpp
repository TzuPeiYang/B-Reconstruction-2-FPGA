#include <TMVA/RModelParser_ONNX.hxx>
#include <TMVA/RModel.hxx>
#include <iostream>
#include <string>

// Global function for ROOT interactive use
void generate_sofie(const std::string& model_path, const std::string& output_dir = "sofie_output") {
    using namespace TMVA::Experimental::SOFIE;

    std::cout << "Starting SOFIE parsing of: " << model_path << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;

    RModelParser_ONNX parser;

    try {
        // Parse ONNX model (this throws std::runtime_error on failure, e.g., unsupported types/ops)
        std::cout << "Parsing model..." << std::endl;
        RModel model = parser.Parse(model_path);
        std::cout << "Parsing completed without errors." << std::endl;

        // Generate C++ code
        std::cout << "Generating C++ code..." << std::endl;
        model.Generate();
        model.OutputGenerated(output_dir);

        std::cout << "C++ code generated successfully in " << output_dir << " (Model.hxx and Model.dat)" << std::endl;
        std::cout << "Next: Compile and run inference.cpp to test." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Parsing failed with runtime error: " << e.what() << std::endl;
        std::cerr << "Tips: Check tensor types (e.g., ReLU input must be float32), shapes, or unsupported ops in edge_convs.0." << std::endl;
        std::cerr << "Run with gDebug=1 for more details." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected parsing error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown parsing error occurred." << std::endl;
    }
}