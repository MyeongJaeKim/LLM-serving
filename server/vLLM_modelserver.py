from abstract_modelserver import AbstractModelServer
from timer_decorator import timer, async_timer

import os

from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.utils import random_uuid

class vLLM(AbstractModelServer):
    def __init__(self,
                 model_id, max_tokens, temperature, top_p,
                 stream_mode = True, 
                 gpu_specified = False, gpu_idx = 0,
                 dtype='auto') -> None:
        # Call super class constructor
        AbstractModelServer.__init__(self, max_tokens, temperature, top_p)   

        # Specifying which GPU to be used
        # * TEST) Tensor parallelism
        if gpu_specified:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        # Sampling config
        self.sampling_params = SamplingParams(temperature = self.temperature,
                                              top_p = self.top_p,
                                              max_tokens = self.max_tokens)
        
        # Model hosting engine
        if stream_mode:
            # Streaming engine
            engine_args = AsyncEngineArgs(
                model = model_id,
                dtype = dtype,
                max_num_seqs = 512,
                # * Tensor parallelism test
                # See : https://docs.vllm.ai/en/latest/serving/distributed_serving.html
                #tensor_parallel_size=4
            )
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            # Non-streaming(blocking) engine
            # self.llm_engine = LLM(model_id, dtype=dtype)

            # Test TensorParallelism
            self.llm_engine = LLM(model_id, dtype=dtype,
                                #   tensor_parallel_size=4
                                  )

        self.stream_mode = stream_mode


    # Override non-streaming inference
    # LLM() class --> for offline batch inference (intended by devlopers)
    @timer
    def inference(self, prompt: str):
        if self.stream_mode == True:
            # Make sure return two values for timer decorator (0 is num_tokens)
            return "Server was loaded as stream mode. Please use stream API.", 0

        outputs = self.llm_engine.generate(prompt, self.sampling_params)

        text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)

        return text, num_tokens

    # Override streaming inference
    # AsyncLLMEngine() --> for class online inferences
    @async_timer
    async def stream_inference(self, prompt: str):
        if (self.stream_mode != True):
            yield "Server was loaded as sync mode. Please use non-stream API."
            # To stop this generator
            return

        generator = self.llm_engine.generate(
            prompt,
            self.sampling_params,
            random_uuid()
        )

        index, num_tokens = 0, 0

        # 1 loop for 1 token
        async for output in generator:
            if (output.outputs[0].text and "\ufffd" == output.outputs[0].text[-1]):
                continue

            #print(output)
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            
            # Comment out when you need
            # num_tokens += 1

            yield text_delta