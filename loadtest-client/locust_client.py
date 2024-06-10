from locust import FastHttpUser, TaskSet, task, between
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP
from locust.env import Environment
from locust.log import setup_logging
import gevent
import time

from prompt_provider import PromptProvider

# Prompt 데이터
prompt_provier = PromptProvider('./filtered_truthfulqa_data.csv')

class ChattingUser(FastHttpUser):
    wait_time = [1, 3]
    host = "http://localhost:8080"

    @task
    def ask_llm(self):
        prompt = prompt_provier.select_question()

        self.client.post(
            url = '/inference/stream',
            json= {'prompt': prompt}
        )


