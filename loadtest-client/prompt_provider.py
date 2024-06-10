import pandas as pd
import random

class PromptProvider():
    def __init__(self, datafile_path: str) -> None:
        self.dataset = pd.read_csv(datafile_path)
        self.data_count = self.dataset.shape[0]
    
    def select_question(self) -> str:
        return self.__get_random_quetion()
    
    def __get_random_quetion(self) -> str:
        index = random.randint(0, self.data_count - 1)
        return self.dataset.iat[index, 0]