# PyTorch 구조 개념 정리

Python 기반 딥러닝 프레임워크 **PyTorch**의 주요 개념 및 구성 요소를 정리한 문서입니다.

---

## 1. 개요 (Overview)

**PyTorch**는 동적 계산 그래프(Dynamic Computational Graph)를 기반으로 한 딥러닝 프레임워크입니다.  
직관적인 코드 작성이 가능하며, 디버깅이 쉬워 연구 및 실무 모두에 적합합니다.

주요 특징:
- 텐서(Tensor) 기반의 계산
- 자동 미분(Autograd)
- 신경망 구성 및 최적화 기능 포함
- Pythonic한 문법

---

## 2. PyTorch란?

- Facebook에서 2017년 발표한 프레임워크로, Lua 기반의 Torch를 Python으로 재구현한 것
- 초기는 과학 연산용 라이브러리였으며, 이후 GPU 기반 딥러닝 프레임워크로 발전
- 유연한 신경망 구축과 빠른 계산 성능 제공

---

## 3. PyTorch의 강점이 두드러지는 분야

- **자연어 처리 (NLP)**  
  → RNN, Transformer 등 복잡한 모델을 유연하게 구현 가능  
- **컴퓨터 비전 (CV)**  
  → 이미지 분류, 객체 탐지, 세분화 등 다양한 비전 모델 구현  
- **강화학습 (RL)**  
  → 동적 그래프 기반의 복잡한 에이전트-환경 인터랙션 처리 용이  
- **연구 및 프로토타이핑**  
  → 직관적이고 디버깅이 쉬워 빠른 실험 가능  
- **커스터마이징이 필요한 모델 개발**  
  → 연산 단위 제어 가능 (낮은 수준까지 접근 가능)

---

## 4. Autograd: 자동 미분 시스템

PyTorch는 연산 이력을 추적하여 자동으로 그래디언트를 계산합니다.

주요 특징:
- 연산 기록은 동적으로 그래프 형태로 저장됨
- `backward()` 호출 시, 연결된 연산을 따라 자동으로 미분 계산
- `requires_grad=True`로 설정된 Tensor에만 미분 적용
- `.grad` 속성으로 최종 기울기(gradient) 확인 가능

---

## 5. PyTorch 주요 구성 요소

| 구성 요소 | 설명 |
|-----------|------|
| `torch` | 텐서 및 기본 수학 함수 제공 |
| `torch.autograd` | 자동 미분 기능 담당 |
| `torch.nn` | 신경망 모듈 제공 (레이어, 손실 함수 등) |
| `torch.optim` | 최적화 알고리즘 (SGD 등) |
| `torch.utils` | 데이터 로딩, 유틸리티 함수 제공 |
| `torch.multiprocessing` | 병렬 처리 기능 |
| `torch.onnx` | ONNX 모델 변환 기능 (타 프레임워크와 호환 가능) |

---

## 6. Tensor 개념 및 특징

- PyTorch에서 모든 데이터는 **Tensor**로 표현됨
- `NumPy`의 `ndarray`와 유사하지만, GPU 연산 지원
- 다양한 차원의 데이터를 표현 가능
- 텐서 간 브로드캐스팅, 슬라이싱, 인덱싱, 조건 연산 등 지원
- GPU 사용을 위한 `.to('cuda')` 또는 `.cuda()` 메서드 제공

---

## 7. Tensor 연산 및 In-place 연산

- 덧셈, 곱셈, 내적, 행렬곱 등 기본 연산 지원
- 브로드캐스팅, 다차원 연산 가능
- 메모리 효율을 위한 `in-place` 연산 지원 (`_`로 끝나는 메서드 사용)

---

## 8. 핵심 메커니즘: 그래디언트 추적 및 전파

- `Tensor.requires_grad = True`로 설정 시 해당 Tensor는 연산 추적 대상이 됨
- 연산 결과 Tensor는 `.grad_fn` 속성을 통해 생성 연산 정보를 가짐
- `.backward()` 호출 시 그래디언트가 연산 그래프를 따라 전파됨
- `.grad` 속성으로 결과 확인 가능

---

## 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [TorchVision](https://pytorch.org/vision/)
- [TorchAudio](https://pytorch.org/audio/)
- [TorchText](https://pytorch.org/text/)

---
## History
작성일: `2025-05-14`