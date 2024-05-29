from abstract_modelserver import AbstractModelServer
from timer_decorator import timer, async_timer

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

class HF_Transformer(AbstractModelServer):
    def __init__(self,
                 model_id_or_path, max_tokens, temperature, top_p,
                 gpu_specified = False, gpu_idx = 0,
                 dtype='auto'):
        AbstractModelServer.__init__(self, max_tokens, temperature, top_p)

        # Generation config
        self.generation_config = {
            'max_length': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': True
        }

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

        # Model hosting
        self.llm_engine = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            #device_map="auto"
            device_map="sequential"
        )

        # For streaming
        self.streamer = TextIteratorStreamer(self.tokenizer,
                                             skip_prompt=True, skip_special_tokens=True)

    # Tokenization wrapper
    def __convert_to_token(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to('cuda')

        return input_ids

    # Override non-stream method
    @timer
    def inference(self, prompt):
        input_ids = self.__convert_to_token(prompt)
        output = self.llm_engine.generate(input_ids, **self.generation_config, use_cache=False)

        text = ''

        token_list = enumerate(output[0])

        for token in token_list:
            text += self.tokenizer.decode(token)

        return text, len(output[0])


    # Override stream method
    @async_timer
    async def stream_inference(self, prompt):
        input_ids = self.__convert_to_token(prompt)

        output = self.llm_engine.generate(input_ids,
                                   **self.generation_config,
                                   streamer=self.streamer,
                                   use_cache=False)

        # full_text = ""

        # NOTE
        # self.streamer --> 'TextIteratorStreamer'.
        # ? TextIteratorStreamer is iterator, __atier__ may be not supported (performance drop on FastAPI side)

        for new_word in self.streamer:
            # Decoding(token id to vocabulary) is not needed
            # full_text += new_word
            yield new_word


