# OpenAI Compatible Server
vLLM 은 SDK 처럼 라이브러리로 프로그램 코드에 로드해서 쓸 수 있는 기능을 제공하고 있는데, 자체 내장 코드로 OpenAI REST API 와 똑같은 인터페이스를 제공하는 자체 서버를 띄울 수 있습니다.

# 전제
vLLM 을 사용하기 위하여 다음과 같은 전제 사항 확인이 필요합니다.
## 버전
* Python : 3.10.x
* CUDA : 12.x
## 모델
### 자체 개발 모델
vLLM 은 지원하는 모델의 종류가 정해져 있으며 버전이 올라갈 때마다 새롭게 발표된 LLM 들을 추가 지원합니다. 다음 링크에서 지원 모델을 확인 할 수 있습니다. 자체 개발한 모델이 아주 새로운 모델이라면 vLLM 에서 지원하기를 기다려야 하지만, Llama2 와 같이 이미 발표된 오픈소스 모델을 기반으로 파인 튜닝한 것이라면 vLLM 지원 목록에서 확인할 수 있을 것이며 vLLM 으로 서빙이 가능합니다.

[지원 모델 목록](https://docs.vllm.ai/en/stable/models/supported_models.html)

자체 개발한 모델이 지원되는지 확인하였다면 모델이 서버에 저장된 directory path 를 확인합니다.

### 허깅페이스 모델
허깅페이스 모델도 마찬가지로 지원 목록에 존재하는지 확인하여야 하며, 그렇지 않은 경우 지원을 기다리거나 base 모델이 지원 목록에 존재하는지 확인하면 됩니다.

그 후 허깅페이스 모델 ID 를 체크하고 아래 가이드에 나올 쉘 스크립트에 기입 합니다. 이미 발표된 모델 중 일부는 허깅페이스 웹 사이트에서 약관 동의가 필요할 수 있고 그런 경우 쉘 환경변수에 허깅페이스 액세스 토큰을 등록할 필요가 있습니다. 관련 가이드는 허깅페이스 내에서 확인하시기 바랍니다.

# vLLM 설치
## 가상환경 만들기 및 활성화
```bash
python3 -m venv .venv
. .venv/bin/activate
```
## vLLM 설치
```bash
pip install vllm
```
## 서버 실행 스크립트(bash) 작성
```bash
# GPU idx
export CUDA_VISIBLE_DEVICES=2

python -m vllm.entrypoints.openai.api_server \
       --model directorypath_or_model_id \
       --dtype float16 \
       --api-key vllm-apikey
```
* GPU 인덱스 : 정책 상 특정 GPU 를 사용하여야 한다면 CUDA_VISIBLE_DEVICES 에 GPU 인덱스를 기록합니다.
* --model : 모델이 존재하는 디렉토리 절대 경로 혹은 허깅페이스 ID 를 작성합니다.
* --dtype : GPU 에서 지원하는 실수 형을 기록합니다. default 는 bfloat16 이므로 GPU 가 bfloat16 을 지원하는 경우 삭제하여도 됩니다.
* --api-key : REST API 의 인증에 사용할 API key 를 넣습니다. API key 인증을 사용하지 않으려면 삭제하여도 됩니다.

이외 다양한 최적화 및 설정 옵션은 다음 링크에서 확인할 수 있습니다.

[기타 옵션 확인](https://docs.vllm.ai/en/stable/models/engine_args.html)

Prefix caching, tensor parallelism 등은 사용 시나리오에 따라 지정할 경우 성능 향상을 기대할 수 있습니다.

## Completion API 테스트 예제

```bash
curl http://localhost:8000/v1/completions \
                -H "Content-Type: application/json;api_key=vllm_apikey" \
                -d '{
                  "model": "directorypath_or_model_id",
                  "prompt": "San Francisco is a",
                  "max_tokens": 512,
                  "temperature": 0.9
                }'
```
* Payload 부분의 model 에는 서버를 실행할 때 사용한 directory path 또는 허깅페이스 모델 ID 를 작성하도록 합니다.
* API key 를 사용하지 않는다면 header 의 api_key 부분을 삭제할 수 있습니다.
* cURL 이 서버에 없다면 yum, dpkg 등의 모듈 관리자로 설치하시기 바랍니다.
* temperature, top_p 등 하이퍼파라미터는 OpenAI 규격에 따라 지정하면 됩니다.

## Chat API 테스트 예제

```bash
curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json;api_key=vllm_apikey" \
        -d '{
                "model": "directorypath_or_model_id",
                "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
                ]
        }'
```
* Payload 부분의 model 에는 서버를 실행할 때 사용한 directory path 또는 허깅페이스 모델 ID 를 작성하도록 합니다.
* API key 를 사용하지 않는다면 header 의 api_key 부분을 삭제할 수 있습니다.
* cURL 이 서버에 없다면 yum, dpkg 등의 모듈 관리자로 설치하시기 바랍니다.

## 클라이언트 코드 작성

OpenAI python SDK 로 클라이언트를 작성하고, 서버 엔드포인트를 실행한 서버로 지정하면 OpenAI 과금을 피하면서 OpenAI 를 backend 로 둔 서비스를 개발/테스트 할 수 있습니다. 자세한 내용은 링크를 확인합니다.

[클라이언트 코드 예제](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#using-openai-chat-api-with-vllm)
