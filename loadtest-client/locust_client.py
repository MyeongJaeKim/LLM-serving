from locust import task, events, FastHttpUser, between
from locust.env import Environment
from locust.log import setup_logging
from locust.stats import stats_history, stats_printer
import gevent

import requests
import json

from file_logger import log

from prompt_provider import PromptProvider
from userid_provider import UserIdProvider

# Prompt 데이터
prompt_provier = PromptProvider("./filtered_truthfulqa_data.csv")
# Random  user Id generator
userid_provider = UserIdProvider()

# -------------------------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------------------------

# Number of users
start_user_count = 1
max_user_count = 1000

# test timings
time_rampup_period_min = 5
time_test_period_min = 5
time_shutdown_period_min = 5

# User 초당 생성 비율
spawn_rate = max_user_count / (time_rampup_period_min * 60)

# 전체 테스트 길이 (램프업 대기 + 10분 간 유지)
total_testlength_min = (time_rampup_period_min + time_test_period_min) * 60

# 테스트 종료 시점에 running user 들이 요청을 마치도록 기다리는 시간
graceful_shutdown_timeout = time_shutdown_period_min * 60


# -------------------------------------------------------------------------------------
# Definition of each load
# -------------------------------------------------------------------------------------
class ChattingUser(FastHttpUser):
    wait_time = between(1, 2)
    # host = "http://localhost:8080"
    # DGX
    # host = "http://50.1.104.14:8000"
    # host = "http://50.1.104.14:18080"
    # H100
    host = "http://10.1.3.142:18080"

    url = host + "/inference/stream"
    # url = host + '/v1/completions'
    weight = 1

    headers = {"Content-Type": "application/json"}

    @task
    def ask_llm(self):
        prompt: str = prompt_provier.select_question()
        userId: str = userid_provider.generate()

        print(f"Current running users : {runner.user_count}")
        log(userId, f"CurrentUserCount || {runner.user_count}")
        log(userId, f"Start to inference || Prompt : {prompt}")

        # for FastAPI
        request_body = {"prompt": prompt}

        # for vLLM OpenAI-compat
        # request_body = {
        #     "model": "/data1/gptL_7B_r240103",
        #     "prompt": prompt,
        #     "max_tokens": 500,
        #     "temperature": 0.9,
        #     "stream": True
        # }

        response = requests.post(
            self.url, headers=self.headers, json=request_body, stream=True
        )

        text: str = ""

        for chunk in response.iter_content(chunk_size=None):
            data: str = chunk.decode("utf-8")

            # --------------------------------
            # for FastAPI
            # --------------------------------
            # token = data.strip()
            # Escape new line
            token = data.replace("\n", " ")

            # --------------------------------
            # for vLLM OpenAI-compat
            # --------------------------------
            # data = data[5:]

            # # End of data stream
            # if '[DONE]' in data.strip():
            #     continue

            # try:
            #     obj = json.loads(data)
            # except:
            #     log(userId, f'Error object {data}')
            #     continue

            # token = obj['choices'][0]['text']
            # # 개행 제거
            # token = token.strip()
            text += token

            # --------------------------------

            log(userId, f"Token received : {token}")

        log(userId, f"End of request || Text : {text}")


# -------------------------------------------------------------------------------------

setup_logging("INFO")

env: Environment = Environment(
    user_classes=[ChattingUser], events=events, stop_timeout=graceful_shutdown_timeout
)

runner = env.create_local_runner()

env.stop_timeout

# runner.

# start a WebUI instance
# web_ui = env.create_web_ui("127.0.0.1", 8089)

# execute init event handlers (only really needed if you have registered any)
# web_ui=web_ui)
env.events.init.fire(environment=env, runner=runner)

# Stat 은  내부 클래스 HttpUser, FastHttpUser 를 사용하지 않아 수집되지 않기 때문에 주석으로 둠
# 사용하지 않은 이유는 내부 클래스가 스트리밍 처리를 할 수 없음
# gevent.spawn(stats_printer(env.stats))
# gevent.spawn(stats_history, env.runner)

# start the test
runner.start(max_user_count, spawn_rate=spawn_rate)

gevent.spawn_later(total_testlength_min, runner.quit)

# wait for the greenlets
runner.greenlet.join()

# stop the web server for good measures
# web_ui.stop()
