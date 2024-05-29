import time, types
from functools import wraps

##
# Timer decorator
#
# Report example
#
#   Generated 727 tokens in 19.3s, throughput = 38 tokens/second.
##

def async_timer(method):
    @wraps(method)
    async def wrapper(*args, **kwargs):
        # Record start time
        start = time.monotonic_ns()

        generator = method(*args, **kwargs)

        # Number of tokens generating
        num_tokens = 0
        
        async for stream_value in generator:
            num_tokens += 1
            yield stream_value
        
        # Record end time
        duration_s = (time.monotonic_ns() - start) / 1e9

        # Report
        print (
            f"\n\tGenerated {num_tokens} tokens in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second.\n"
        )
    
    return wrapper

def timer(method):
    @wraps(method)
    def wrapper(*args, **kwargs):

        # Record start time
        start = time.monotonic_ns()

        text, num_tokens = method(*args, **kwargs)

        # Record end time
        duration_s = (time.monotonic_ns() - start) / 1e9

        # Report
        print (
            f"\n\tGenerated {num_tokens} tokens in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second.\n"
        )

        return text, num_tokens
    
    return wrapper
