import random
import string

class UserIdProvider():
    def __init__(self) -> None:
        pass

    def generate(self) -> str:
        # Size of string
        N = 16

        res = ''.join(random.choices(string.ascii_lowercase  + string.digits, k=N))
        
        return res
    