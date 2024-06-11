from locust import task, events, FastHttpUser, between
from locust.env import Environment
from locust.log import setup_logging
from locust.stats import stats_history, stats_printer
import gevent

import requests

from file_logger import log

from prompt_provider import PromptProvider
from userid_provider import UserIdProvider

# Prompt 데이터
prompt_provier = PromptProvider('./filtered_truthfulqa_data.csv')
userid_provider = UserIdProvider()

# -------------------------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------------------------

# Number of users
start_user_count = 1
max_user_count = 10

rampup_user = 1
rampup_interval_sec = 30

# User 초당 생성 비율
spawn_rate = rampup_user / rampup_interval_sec

# 전체 테스트 길이
test_length_in_min = 10 * 60

# -------------------------------------------------------------------------------------
# Definition of each load
# -------------------------------------------------------------------------------------
class ChattingUser(FastHttpUser):
    wait_time = between(1, 2)
    host = "http://localhost:8080"
    url = host + '/inference/stream'
    weight = 1

    headers = {
        'Content-Type': 'application/json'
    }

    @task
    def ask_llm(self):
        prompt: str = prompt_provier.select_question()
        userId: str = userid_provider.generate()

        log(userId, f"Start to inference || Prompt : {prompt}")

        request_body = {
            'prompt': prompt
        }

        response = requests.post(self.url, headers=self.headers, json=request_body, stream=True)

        for chunk in response.iter_content(chunk_size=None):
            token: str = chunk.decode()
            
            log(userId, f'Token received : {token}')
        
        log(userId, f'End of request || Prompt : {prompt}')


# -------------------------------------------------------------------------------------

setup_logging("INFO")

env: Environment = Environment(user_classes=[ChattingUser], events=events)

runner = env.create_local_runner()

# start a WebUI instance
web_ui = env.create_web_ui("127.0.0.1", 8089)

# execute init event handlers (only really needed if you have registered any)
env.events.init.fire(environment=env, runner=runner, web_ui=web_ui)

# start a greenlet that periodically outputs the current stats
# gevent.spawn(stats_printer(env.stats))

# start a greenlet that save current stats to history
gevent.spawn(stats_history, env.runner)

# start the test
runner.start(max_user_count, spawn_rate=spawn_rate)

gevent.spawn_later(test_length_in_min, runner.quit)

# wait for the greenlets
runner.greenlet.join()

# stop the web server for good measures
web_ui.stop()

