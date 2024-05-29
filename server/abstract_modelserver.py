from abc import ABC, abstractmethod


class AbstractModelServer(ABC):
    def __init__(self, max_tokens, temperature, top_p) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def inference(self, prompts):
        pass

    @abstractmethod
    def stream_inference(self, prompts):
        pass
