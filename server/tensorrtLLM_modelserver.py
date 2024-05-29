from abstract_modelserver import AbstractModelServer

import time

from tensorrt_llm import LLM, ModelConfig
from tensorrt_llm.hlapi.tokenizer import TokenizerBase, TransformersTokenizer

# https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/high-level-api

class TensorRTLLM(AbstractModelServer):
    def __init__(self, converted_model_path, origin_model_path,  max_tokens, temperature, top_p):
        # Call super class constructor
        AbstractModelServer.__init__(self, max_tokens, temperature, top_p)

        model_config = ModelConfig(model_dir = converted_model_path)

        # Tokenizer
        # 컨버팅 후 토크나이저가 안 나타나서, 수동으로 바인딩
        self.tokenizer = TransformersTokenizer.from_pretrained(pretrained_model_dir = origin_model_path)

        # Async LLM
        #
        self.async_llm = LLM(model_config, streaming_llm = True , tokenizer=self.tokenizer)

        print(self.async_llm)

        self.sampling_config = self.async_llm.get_default_sampling_config()
        self.sampling_config.temperature = [self.temperature]
        # top_p, top_k 는 아직 지원되지 않는다.
        #self.sampling_config.top_p = [self.top_p]

    # Override stream
    async def stream_inference(self, prompt: str):
        # Performance measure
        index, num_tokens = 0, 0
        start = time.monotonic_ns()

        async for output in self.async_llm.generate_async(prompt,
                                                          self.sampling_config,
                                                          streaming = True):
            text_delta = output.text[index:]
            print(text_delta)
            num_tokens = len(output.token_ids)
            index = len(output.text)
            yield text_delta
        
        # Performance Report
        duration_s = (time.monotonic_ns() - start) / 1e9

        print (
            f"\n\tGenerated {num_tokens} tokens in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second.\n"
        )

    # Override non-stream inference
    def inference(self, prompts):
        pass