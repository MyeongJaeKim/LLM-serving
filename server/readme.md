# Notice
## 리포지터리의 목적
* 대규모 언어 모델(LLM)을 실제 서비스(live) 환경에서 사용하려면 모델을 로드해주는 서빙 라이브러리가 필요합니다. 다음 서빙 라이브러리들을 사용하여 모델을 로드하고 REST API 인터페이스를 제공하는 코드를 저장소에서 제공합니다.
  * Transformers (huggingface)
  * vLLM (UC Berkely)
  * TensorRT-LLM (NVIDIA)
* 서빙 라이브러리 별로 하이퍼 파라미터와 성능 최적화 파라미터 지정 기능, 스트리밍 및 블러킹 인터페이스를 동시에 제공합니다.

## 버전
코드를 정상적으로 실행하기 위해서 다음 사항을 점검하시기 바랍니다.
* Python 버전 : 3.10.x
* CUDA : 12.x

## 시스템 라이브러리 설치
### TensorRT-LLM
```bash
apt-get -y install openmpi-bin libopenmpi-dev
```

# 환경 설정
vLLM 과 TensorRT-LLM 은 같은 라이브러리의 다른 버전을 사용합니다. 그래서 같은 가상환경에서 모든 의존성 라이브러리를 설치할 수 없습니다. 다음을 참고하여 가상환경을 설정하시기 바랍니다.
## 가상환경 만들기
### vLLM
```bash
python3 -m venv .venv_vllm
```
### TensorRT-LLM
```bash
python3 -m vevn .venv_tensorrtllm
```
### Transformers
```bash
python3 -m venv .venv_transformers
```
## 가상환경 사용하는 법
### 사용을 시작할 때
```bash
. .venv_vllm/bin/activate
```
### 사용을 끝낼 때
```bash
deactivate
```

## 의존성 라이브러리 설치
가상환경을 activate 한 상태에서, 다음과 같이 의존성 라이브러리를 설치합니다.
```bash
pip install -r requirements_vllm.txt
```
### 예시
```bash
(.venv_vllm) ~/Test$ pip install -r requirements_vllm.txt
```

# 기타 사항
## 아직 지원되지 않는 것들
* FastAPI 서버를 구동할 때 서빙 라이브러리 중 1개만 선택할 수 있습니다.
* 파라미터화 되어 있지 않아 코드 수정이 필요합니다.
```python
# -------------------------
# LLM serving engine
# ! Each dependent libraries of vLLM, TensorRT-LLM are different
# -------------------------
# vLLM
llm_engine = vLLM(model_id, max_tokens, temperature, top_p,
                  stream_mode=False,
                  gpu_specified=gpu_specified, gpu_idx=gpu_idx,
                  dtype=dtype)

# TensorRT-LLM
#llm_engine = TensorRTLLM(converted_model_path, origin_model_path, max_tokens, temperature, top_p)

# Transformers (huggingface)
# CURRENTLY NOT AVAILABLE - GPU specifying (gpu idx)
# llm_engine = HF_Transformer(model_id, max_tokens, temperature, top_p,
                            # dtype=dtype)
```

