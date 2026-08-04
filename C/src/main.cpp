#include <stdio.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>

#include "particlenet_complete.h"
#include "pf_points.h"
#include "pf_features.h"
#include "pf_mask.h"

static const int N_SAMPLES    = 2000;
static const int N_PT_CH      = 3;   // pf_points channels
static const int N_FEAT_CH    = 4;   // pf_features channels
static const int N_PARTICLES  = 35;


// Storage for one sample
struct Sample {
    float pf_points  [1][N_PT_CH  ][N_PARTICLES];
    float pf_features[1][N_FEAT_CH][N_PARTICLES];
    float pf_mask    [1][1        ][N_PARTICLES];
};

// ---------------------------------------------------------------------------
// Parse the text file produced by calibration.npz → calibration.txt
// Returns number of samples actually read.
// ---------------------------------------------------------------------------
static int load_calibration(const char* path, Sample samples[N_SAMPLES])
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[TB] ERROR: cannot open " << path << "\n";
        return -1;
    }

    int n_read = 0;
    std::string line;

    while (std::getline(fin, line)) {
        // Skip comments and blank lines
        if (line.empty() || line[0] == '#') continue;

        if (line.rfind("SAMPLE ", 0) == 0) {
            int idx = std::atoi(line.c_str() + 7);
            if (idx < 0 || idx >= N_SAMPLES) {
                std::cerr << "[TB] WARNING: sample index " << idx
                          << " out of range, skipping.\n";
                continue;
            }
            Sample& s = samples[idx];

            // --- PF_POINTS ---
            std::getline(fin, line); // "PF_POINTS"
            for (int c = 0; c < N_PT_CH; ++c) {
                std::getline(fin, line);
                std::istringstream ss(line);
                for (int p = 0; p < N_PARTICLES; ++p)
                    ss >> s.pf_points[0][c][p];
            }

            // --- PF_FEATURES ---
            std::getline(fin, line); // "PF_FEATURES"
            for (int c = 0; c < N_FEAT_CH; ++c) {
                std::getline(fin, line);
                std::istringstream ss(line);
                for (int p = 0; p < N_PARTICLES; ++p)
                    ss >> s.pf_features[0][c][p];
            }

            // --- PF_MASK ---
            std::getline(fin, line); // "PF_MASK"
            std::getline(fin, line);
            {
                std::istringstream ss(line);
                for (int p = 0; p < N_PARTICLES; ++p)
                    ss >> s.pf_mask[0][0][p];
            }

            ++n_read;
        }
    }

    fin.close();
    std::cout << "[TB] Loaded " << n_read << " samples from " << path << "\n";
    return n_read;
}


// ---------------------------------------------------------------------------
// Main testbench
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    const char* data_file = (argc > 1) ? argv[1] : "./calibration.txt";

    // Allocate sample storage (stack is fine for 500 samples at this size)
    static Sample samples[N_SAMPLES];

    int n = load_calibration(data_file, samples);
    if (n <= 0) return 1;


    std::ofstream outputFile("./result.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        // ----------------------------------------------------------------
        // Pack inputs into whatever array/stream type your DUT expects.
        // The example below uses plain arrays — adapt to hls::stream<> or
        // ap_fixed<> as required by your design.
        // ----------------------------------------------------------------

        float in_points  [1][N_PT_CH  ][N_PARTICLES];
        float in_features[1][N_FEAT_CH][N_PARTICLES];
        float in_mask    [1][1        ][N_PARTICLES];

        for (int c = 0; c < N_PT_CH;   ++c)
            for (int p = 0; p < N_PARTICLES; ++p)
                in_points[0][c][p] = samples[i].pf_points[0][c][p];

        for (int c = 0; c < N_FEAT_CH; ++c)
            for (int p = 0; p < N_PARTICLES; ++p)
                in_features[0][c][p] = samples[i].pf_features[0][c][p];

        for (int p = 0; p < N_PARTICLES; ++p)
            in_mask[0][0][p] = samples[i].pf_mask[0][0][p];

        // ----------------------------------------------------------------
        // DUT call — replace with your actual function signature
        // ----------------------------------------------------------------
        float tensor_MSE[1][4]; // placeholder output
        entry(in_points, in_features, in_mask, tensor_MSE);

        for (int i = 0; i < 4; i++) {
            outputFile << tensor_MSE[0][i] << " ";
        }
        outputFile << std::endl;
        // ----------------------------------------------------------------
        // Check result
        // ----------------------------------------------------------------
        
    }
    outputFile.close();
    return 0;
}
