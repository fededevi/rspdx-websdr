#!/usr/bin/env python3
"""Whisper transcription service for a computer with an NVIDIA GPU.

POST /transcribe with 16-bit mono PCM at 8 kHz as the body. Query options:
  prompt=<text>  the initial prompt (optional)
  lang=<code>    the language (optional, default: detect)
The answer is JSON: {"language", "probability", "segments": [{"start", "end", "text",
"avg_logprob", "no_speech_prob", "compression_ratio"}], "seconds"}.

GET /health gives the model and the device.

It listens on 127.0.0.1 only. Reach it through an SSH port forward.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

p = argparse.ArgumentParser()
p.add_argument("--model", default="large-v3-turbo", help="a model name, or the folder of a downloaded model")
p.add_argument("--device", default="cuda")
p.add_argument("--compute-type", default="int8", help="int8 for Pascal GPUs, which have slow float16")
p.add_argument("--port", type=int, default=8491)
args = p.parse_args()

# the CUDA libraries come from the nvidia-* Python packages
if sys.platform == "win32":
    import site
    for base in site.getsitepackages():
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            d = os.path.join(base, "nvidia", sub, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from scipy import signal  # noqa: E402

t0 = time.time()
model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
print(f"{args.model} on {args.device} ({args.compute_type}) loaded in {time.time() - t0:.0f} s", flush=True)
gpu = threading.Lock()   # one transcription at a time on the GPU


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self.send(200, {"model": args.model, "device": args.device, "compute_type": args.compute_type})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/transcribe":
            self.send(404, {"error": "not found"})
            return
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        n = int(self.headers.get("Content-Length", 0))
        pcm = np.frombuffer(self.rfile.read(n), "<i2")
        audio = signal.resample_poly(pcm.astype(np.float32) / 32768, 2, 1)
        t = time.time()
        with gpu:
            parts, info = model.transcribe(audio, language=qs.get("lang") or None, beam_size=1, vad_filter=True,
                                           condition_on_previous_text=False, initial_prompt=qs.get("prompt") or None)
            parts = list(parts)
        self.send(200, {"language": info.language, "probability": info.language_probability,
                        "seconds": round(time.time() - t, 2),
                        "segments": [{"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                                      "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio}
                                     for s in parts]})


srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
print(f"listening on 127.0.0.1:{args.port}", flush=True)
srv.serve_forever()
