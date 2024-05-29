from vLLM_modelserver import vLLM
#from tensorrtLLM_modelserver import TensorRTLLM
import torch
# from transformer_modelserver import HF_Transformer

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# -------------------------
# To read the structure of request body
from pydantic import BaseModel

class UserPrompt(BaseModel):
    prompt: str
# -------------------------

app = FastAPI()

# -------------------------
# Parameters
# -------------------------
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#model_id = "meta-llama/Meta-Llama-3-8B"
#model_id = "mistralai/Mistral-7B-v0.1"
temperature = 0.8
top_p = 0.95
max_tokens = 1024

# only for TensorRT-LLM
converted_model_path = "TRT-LLM/mistral-7b-1gpu-bf16"
origin_model_path = "TRT-LLM/Mistral-7B-Instruct-v0.2"

# -------------------------
# GPU dependent settings
# -------------------------
gpu_specified = True

# gpu_specified == True : then use the 'gpu_idx'
gpu_idx = 0

# DGX only supports dtype=torch.float16

# vLLM
dtype = 'float16'
# Transformers
# dtype = torch.float16

# GCP
# dtype = 'auto' # default value (bfloat16)

# -------------------------
# LLM serving engine
# ! Each dependent libraries of vLLM, TensorRT-LLM are different
# -------------------------
# vLLM
llm_engine = vLLM(model_id, max_tokens, temperature, top_p,
                  stream_mode=False,
                  gpu_specified=gpu_specified, gpu_idx=gpu_idx,
                  dtype=dtype)

# TensorRT-LLM
#llm_engine = TensorRTLLM(converted_model_path, origin_model_path, max_tokens, temperature, top_p)

# Transformers (huggingface)
# CURRENTLY NOT AVAILABLE - GPU specifying (gpu idx)
# llm_engine = HF_Transformer(model_id, max_tokens, temperature, top_p,
                            # dtype=dtype)

# -------------------------
# Streaming inference
# -------------------------

# ! FastAPI 는 ASGI. 동시성을 높이기 위해서 async API 는 async call 만 사용해야 함.
# ! await 가 async 호출 스택 내 존재하면 인터프리터 쓰레드 풀을 사용하게 되어, 모든 request 는 blocking 처리됨.

@app.post("/inference/stream")
async def inference_stream(requestBody: UserPrompt):

    user_prompt: str = requestBody.prompt

    async def stream_generation(prompt):
        async for text in llm_engine.stream_inference(prompt):
            yield text
    
    return StreamingResponse(stream_generation(user_prompt), media_type="text/event-stream")

# -------------------------
# Blocking inference (non-streaming)
# -------------------------
@app.post("/inference/nonstream")
async def inference_nonstream(requestBody: UserPrompt):

    user_prompt: str = requestBody.prompt

    text, tokens = llm_engine.inference(user_prompt)
    
    return text
