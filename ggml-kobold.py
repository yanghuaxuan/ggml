# Supa scuffed script to make ggml models work with Kobold
# Inspired by https://github.com/LostRuins/llamacpp-for-kobold

import ctypes
import os
from pathlib import Path
import json, http.server, threading, socket, sys, time
import re

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
libgptj.generate.argtypes = [gpt_params_c, ctypes.POINTER(ctypes.c_char_p)] 
libgptj.generate.restype = ctypes.c_int;
   
def load_model(parameters):
    return libgptj.load_model(parameters)

def generate(parameters, p_prompt_p):
    return libgptj.generate(parameters, p_prompt_p)

#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
friendlymodelname = "concedo/llamacpp"  # local kobold api apparently needs a hardcoded known HF model name
maxctx = 2048
maxlen = 128
modelbusy = False

class ServerRequestHandler(http.server.SimpleHTTPRequestHandler):
    sys_version = ""
    server_version = "ConcedoLlamaForKoboldServer"

    def __init__(self, addr, port, embedded_kailite):
        self.addr = addr
        self.port = port
        self.embedded_kailite = embedded_kailite

    def __call__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        global maxctx, maxlen, friendlymodelname
        if self.path in ["/", "/?"] or self.path.startswith(('/?','?')): #it's possible for the root url to have ?params without /
            response_body = ""
            if self.embedded_kailite is None:
                response_body = (f"Embedded Kobold Lite is not found.<br>You will have to connect via the main KoboldAI client, or <a href='https://lite.koboldai.net?local=1&port={self.port}'>use this URL</a> to connect.").encode()
            else:
                response_body = self.embedded_kailite

            self.send_response(200)
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            return
                       
        self.path = self.path.rstrip('/')
        if self.path.endswith(('/api/v1/model', '/api/latest/model')):
            self.send_response(200)
            self.end_headers()
            result = {'result': friendlymodelname }
            self.wfile.write(json.dumps(result).encode())
            return

        if self.path.endswith(('/api/v1/config/max_length', '/api/latest/config/max_length')):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"value": maxlen}).encode())
            return

        if self.path.endswith(('/api/v1/config/max_context_length', '/api/latest/config/max_context_length')):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"value": maxctx}).encode())
            return

        if self.path.endswith(('/api/v1/config/soft_prompt', '/api/latest/config/soft_prompt')):
            self.send_response(200)
            self.end_headers()           
            self.wfile.write(json.dumps({"value":""}).encode())
            return
        
        self.send_response(404)
        self.end_headers()
        rp = 'Error: HTTP Server is running, but this endpoint does not exist. Please check the URL.'
        self.wfile.write(rp.encode())
        return
    
    def do_POST(self):
        global modelbusy
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)  
        basic_api_flag = False
        kai_api_flag = False
        self.path = self.path.rstrip('/')

        if modelbusy:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(json.dumps({"detail": {
                    "msg": "Server is busy; please try again later.",
                    "type": "service_unavailable",
                }}).encode())
            return

        if self.path.endswith('/request'):
            basic_api_flag = True

        if self.path.endswith(('/api/v1/generate', '/api/latest/generate')):
            kai_api_flag = True

        if basic_api_flag or kai_api_flag:
            genparams = None
            try:
                genparams = json.loads(body)
            except ValueError as e:
                self.send_response(503)
                self.end_headers()
                return       
            print("\nInput: " + json.dumps(genparams))
            
            modelbusy = True
            if kai_api_flag:
                fullprompt = genparams.get('prompt', "")
            else:
                fullprompt = genparams.get('text', "")
            newprompt = fullprompt
            
            recvtxt = ""
            output = ctypes.c_char_p(b"")
            if kai_api_flag:
                parameters.seed=-1
                parameters.prompt=newprompt.encode("ascii")
                #parameters.max_context_length = ...
                parameters.temp = float(genparams.get('temperature', parameters.temp))
                parameters.top_k = int(genparams.get('top_k', parameters.top_k))
                parameters.top_p= float(genparams.get('top_p', parameters.top_p))
                parameters.n_predict = int(genparams.get('max_length', parameters.n_predict))
                print("PROMPT ===========: " + parameters.prompt.decode("ascii", "ignore"))
                status = generate(parameters, ctypes.byref(output))
                recvtxt = output.value.decode("ascii", "ignore")
                #print("PATTERN ==========: " + "^" + rf"{parameters.prompt.decode('ascii', 'ignore')}")
                recvtxt = re.sub("^" + rf"{re.escape(parameters.prompt.decode('ascii', 'ignore'))}", "", recvtxt)

                print("\nOutput: " + recvtxt)
                res = {"results": [{"text": recvtxt}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())               
            else:
                parameters.seed=-1
                parameters.prompt=newprompt.encode("ascii")
                #parameters.max_context_length = ...
                parameters.temp = float(genparams.get('temperature', parameters.temp))
                parameters.top_k = int(genparams.get('top-k', parameters.top_k))
                parameters.n_predict = int(genparams.get('max_length', parameters.n_predict))
                parameters.top_p= float(genparams.get('top_p', parameters.top_p))

                status = generate(parameters, ctypes.POINTER(recvtxt))
                recvtxt.decode("ascii", "ignore")
                #print("PATTERN ==========: " + "^" + rf"{parameters.prompt.decode('ascii', 'ignore')}")
                recvtxt = re.sub("^" + rf"{parameters.prompt.decode('ascii', 'ignore')}", "", recvtxt)
                print("\nOutput: " + recvtxt)
                res = {"data": {"seqs":[recvtxt]}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())
            modelbusy = False
            return    
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        if "/api" in self.path:
            self.send_header('Content-type', 'application/json')
        else:
            self.send_header('Content-type', 'text/html')
           
        return super(ServerRequestHandler, self).end_headers()


def RunServerMultiThreaded(addr, port, embedded_kailite = None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((addr, port))
    sock.listen(5)

    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()

        def run(self):
            handler = ServerRequestHandler(addr, port, embedded_kailite)
            with http.server.HTTPServer((addr, port), handler, False) as self.httpd:
                try:
                    self.httpd.socket = sock
                    self.httpd.server_bind = self.server_close = lambda self: None
                    self.httpd.serve_forever()
                except (KeyboardInterrupt,SystemExit):
                    self.httpd.server_close()
                    sys.exit(0)
                finally:
                    self.httpd.server_close()
                    sys.exit(0)
        def stop(self):
            self.httpd.server_close()

    numThreads = 5
    threadArr = []
    for i in range(numThreads):
        threadArr.append(Thread(i))
    while 1:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            for i in range(numThreads):
                threadArr[i].stop()
            sys.exit(0)

if __name__ == "__main__":
    parameters = gpt_params_c()
    parameters = gpt_params_c()
    parameters.seed = -1 
    parameters.n_threads = 1
    parameters.n_predict =  80
    parameters.top_k = 40
    parameters.top_p = 0.9
    parameters.temp = 1.0
    parameters.n_batch = 8
    # TODO: Add n_ctx tuning
    parameters.model = os.path.join(Path.home(), "text-generation-webui/models/pygmalion-6b_f16/ggml-model.bin").encode("ascii")
    parameters.prompt = "Hello world!".encode("ascii", "ignore")

    args = {"port": 5001, "host": "localhost"}

    ggml_selected_file = parameters.model.decode("ascii")
    embedded_kailite = None 
    if not ggml_selected_file:     
        #give them a chance to pick a file
        print("Please manually select ggml file:")
        from tkinter.filedialog import askopenfilename
        ggml_selected_file = askopenfilename (title="Select ggml model .bin files")
        if not ggml_selected_file:
            print("\nNo ggml model file was selected. Exiting.")
            time.sleep(1)
            sys.exit(2)

    if not os.path.exists(ggml_selected_file):
        print(f"Cannot find model file: {ggml_selected_file}")
        time.sleep(1)
        sys.exit(2)

    mdl_nparts = sum(1 for n in range(1, 9) if os.path.exists(f"{ggml_selected_file}.{n}")) + 1
    modelname = os.path.abspath(ggml_selected_file)
    print(f"Loading model: {modelname}, Parts: {mdl_nparts}, Threads: {parameters.n_threads}")
    loadok = load_model(parameters)
    print("Load Model OK: " + str(loadok))

    if loadok != 0:
        print("Could not load model: " + modelname)
        time.sleep(1)
        sys.exit(3)
    try:
        basepath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(basepath, "klite.embd"), mode='rb') as f:
            embedded_kailite = f.read()
            print("Embedded Kobold Lite loaded.")
    except:
        print("Could not find Kobold Lite. Embedded Kobold Lite will not be available.")

    print(f"Starting Kobold HTTP Server on port {args['port']}")
    epurl = ""
    if args["host"]=="":
        epurl = f"http://localhost:{args['port']}" + ("?streaming=1" if not parameters.nostream else "")   
    else:
        epurl = f"http://{args['host']}:{args['port']}?host={args['host']}" #+ ("&streaming=1" if not parameters.nostream else "")   TODO: Add stream
    
        
    print(f"Please connect to custom endpoint at {epurl}")
    RunServerMultiThreaded(args["host"], args["port"], embedded_kailite)
