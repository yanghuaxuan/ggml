# Supa scuffed script to make ggml models work with Kobold
# Inspired by https://github.com/LostRuins/llamacpp-for-kobold

import ctypes
import os
from pathlib import Path

class gpt_params_c(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int32),
                ("n_threads", ctypes.c_int32),
                ("n_predict", ctypes.c_int32),
                ("top_k", ctypes.c_int32),
                ("top_p", ctypes.c_float),
                ("temp", ctypes.c_float),
                ("n_batch", ctypes.c_int32),
                ("model", ctypes.c_char_p),
                ("prompt", ctypes.c_char_p)]

dir_path = os.path.dirname(os.path.realpath(__file__))
libgptj = ctypes.CDLL(dir_path + "/build/examples/gpt-j/liblibgptj.dylib")

libgptj.load_model.argtypes = [gpt_params_c] 
libgptj.load_model.restype = ctypes.c_int;
libgptj.generate.argtypes = [gpt_params_c] 
libgptj.generate.restype = ctypes.c_char_p;
   
def load_model(parameters):
    return libgptj.load_model(parameters)

def generate(parameters):
    parameters.model = "Hello world!".encode("UTF-8")
    return libgptj.generate(parameters)

if __name__ == "__main__":
    parameters = gpt_params_c()
    parameters = gpt_params_c()
    parameters.seed = -1 
    parameters.n_threads = 1
    parameters.n_predict =  200
    parameters.top_k = 40
    parameters.top_p = 0.9
    parameters.temp = 1.0
    parameters.n_batch = 8
    parameters.model = os.path.join(Path.home(), "text-generation-webui/models/pygmalion-6b_f16/ggml-model.bin").encode("UTF-8")
    parameters.prompt = "".encode("UTF-8")

    load_model(parameters)
    print(generate(parameters).decode("UTF-8"))
