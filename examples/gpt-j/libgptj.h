#include "ggml/ggml.h"
#include "utils.h"
#include <vector>
#include <map>
#include <ios>
#include <fstream>
#include <iostream>
#include "unistd.h"
#include "wchar.h"

struct gpt_params_c { int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict = 200; // new tokens to predict

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.9f;
    float   temp  = 1.0f;

    int32_t n_batch = 8; // batch size for prompt processing

    const char * model = "models/gpt-2-117M/ggml-model.bin"; // model path
    const char * prompt;
    
    // TODO: Add n_ctx param
};

// default hparams (GPT-J 6B)
struct gptj_hparams {
    int32_t n_vocab = 50400;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 4096;
    int32_t n_head  = 16;
    int32_t n_layer = 28;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct gptj_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    // attention
    struct ggml_tensor * c_attn_q_proj_w;
    struct ggml_tensor * c_attn_k_proj_w;
    struct ggml_tensor * c_attn_v_proj_w;

    struct ggml_tensor * c_attn_proj_w;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w_trans;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gptj_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding

    struct ggml_tensor * lmh_g; // language model head
    struct ggml_tensor * lmh_b; // language model bias

    std::vector<gptj_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

bool gptj_model_load(const std::string & fname, gptj_model & model, gpt_vocab & vocab);


bool gptj_eval(
        const gptj_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token);

extern gpt_vocab vocab;
extern gptj_model model;

// C function to use for our Python KAI
#ifdef __cplusplus
extern "C" int load_model(gpt_params_c * params);
#else
int load_model(gpt_params_c * params);
#endif
// C function to use for our Python KAI
#ifdef __cplusplus
extern "C" const char * generate(gpt_params_c & params);
#else
const char * generate(gpt_params_c & params);
#endif
