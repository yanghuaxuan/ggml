# ggml for kobold
A port of [llamacpp-for-kobold](https://github.com/LostRuins/llamacpp-for-kobold) for the GGML examples.  
Only GPT-J models, and other models derived from GPT-J (like Pygmalion) are supported right now. 

## Usage
You must first build the GGML library + GPTJ example. How's an example for running on Linux
```
# Build ggml + examples
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j4

# Run KAI server
cd ..
python ggml-kobold.py

# Now you can connect to KoboldAI! The default link is http://localhost:5001
```

## TODO
* Add maximum context tuning for KoboldAI frontend

## License
* The original GGML library and examples by ggerganov are licensed under the MIT License
* The original python script for running KoboldAI is licensed under AGPL v3.0 Licenese
* Kobold Lite is also licensed under the AGPL v3.0 License
* The C bindings are licenesed under the AGPL v3.0 License
