from abstract_modelserver import AbstractModelServer
from timer_decorator import async_timer

import time

from tensorrt_llm import LLM, ModelConfig
from tensorrt_llm.hlapi.utils import SamplingConfig
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

        self.llm_engine = LLM(model_config, streaming_llm = True , tokenizer=self.tokenizer)

    def __calc_new_token_length(self, prompt: str) -> int:
        prompt_token_ids = self.tokenizer.encode(prompt)
        max_new_tokens = self.max_tokens - len(prompt_token_ids)
        print(f'Calculated max_new_tokens : {max_new_tokens}')
        return max_new_tokens

    def __generate_sampling_configs(self, max_new_tokens: int) -> SamplingConfig:
        sampling_config: SamplingConfig = self.llm_engine.get_default_sampling_config()
        sampling_config.temperature = [self.temperature]
        # top_p, top_k 는 아직 지원되지 않는다.
        #sampling_config.top_p = [self.top_p]
        sampling_config.max_new_tokens = max_new_tokens

        return sampling_config


    # Override stream
    @async_timer
    async def stream_inference(self, prompt: str):
        index = 0

        max_new_tokens = self.__calc_new_token_length(prompt)
        sampling_configs = self.__generate_sampling_configs(max_new_tokens)

        async for output in self.llm_engine.generate_async(prompt,
                                                          sampling_configs,
                                                          streaming = True):
            text_delta = output.text[index:]
            num_tokens = len(output.token_ids)
            index = len(output.text)
            yield text_delta

    # Override non-stream inference
    def inference(self, prompts):
        pass