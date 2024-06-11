from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import random
import asyncio

# To parse the structure of request body
from pydantic import BaseModel

class UserPrompt(BaseModel):
    prompt: str

app = FastAPI()

@app.post("/inference/stream")
async def inference_stream(requestBody: UserPrompt, request: Request):

    async def pick_token(prompt):
        async for text in __generate_streaming_random_response():
            print(f'Generated {text}')

            if await request.is_disconnected():
                # Detect client leave and stop sending token
                return

            yield text

    
    return StreamingResponse(pick_token(requestBody.prompt), media_type="text/event-stream")

# LLM 을 모사하는 random generator
async def __generate_streaming_random_response():
    def get_random_number():
        return random.randint(1, 50127)
    
    repeat_count = random.randint(100, 200)

    for _ in range(repeat_count):
        number = get_random_number()

        sleep_time = random.uniform(0.0, 1.0)
        await asyncio.sleep(sleep_time)

        yield str(number)