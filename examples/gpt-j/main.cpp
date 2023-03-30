#include "libgptj.h"
#include <stdint.h>
#include "wchar.h"

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/gpt-j-6B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (!gptj_model_load(params.model, model, vocab)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    gpt_params_parse(argc, argv, params);
    params.prompt.assign("Pepega");

    gpt_params_c params_c = {};
    params_c.seed = params.seed;
    params_c.n_threads = params.n_threads;
    params_c.n_predict = params.n_predict;
    params_c.top_k = params.top_k;
    params_c.top_p = params.top_p;
    params_c.temp = params.temp;
    params_c.n_batch = params.n_batch;
    params_c.model = params.model.c_str();
    params_c.prompt = params.prompt.c_str();

    const char * generated_text = generate(params_c);
    if (generated_text != NULL) {
        std::cout << generate(params_c);
    }

    ggml_free(model.ctx);

    return 0;
}
