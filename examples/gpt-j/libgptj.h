#include "ggml/ggml.h"
#include "utils.h"
#include <vector>
#include <map>
#include <ios>
#include <fstream>
#include <iostream>
#include "unistd.h"
#include "wchar.h"
#include "main.cpp"


struct gpt_params_c { int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict = 20; // new tokens to predict

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.9f;
    float   temp  = 1.0f;

    int32_t n_batch = 8; // batch size for prompt processing

    const char * model = "models/gpt-2-117M/ggml-model.bin"; // model path
    const char * prompt;
    
    // TODO: Add n_ctx param
};

static gptj_hparams hparams;

// C function to use for our Python KAI
extern "C" int load_model(gpt_params_c * params);
// C function to use for our Python KAI
extern "C" int generate(gpt_params_c & params, char ** output);
