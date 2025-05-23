# PyTorch Tensor 개념 및 활용 정리

Python 기반 딥러닝 프레임워크 **PyTorch**의 핵심 구성 요소인 **Tensor**의 개념 및 실습 예제를 정리한 문서입니다.

---

## 1. 개요 (Overview)

**Tensor**는 PyTorch에서 데이터를 표현하는 기본 단위입니다.  
NumPy의 `ndarray`와 유사하지만, GPU 연산을 지원하여 고속 처리가 가능합니다.

PyTorch Tensor의 특징:
- 다양한 방식으로 Tensor 초기화 가능
- GPU 전송을 통한 연산 가속
- 기존 Tensor로부터 속성 복사 생성
- 크기 변경, 브로드캐스팅, 연산 지원

---

## 2. PyTorch란?

- Python 기반 과학 연산 패키지
- 두 가지 주요 용도:
  - GPU 가속이 필요한 NumPy 대체
  - 유연한 딥러닝 연구 플랫폼
- 동적 계산 그래프 기반

---

## 3. Tensor 생성 및 초기화

다양한 방법으로 Tensor를 생성할 수 있습니다:

```python
import torch

# 초기화되지 않은 텐서 (메모리 쓰레기값 포함)
x = torch.empty(5, 3)

# 무작위 초기화
x = torch.rand(5, 3)

# 0으로 초기화된 Long 타입 텐서
x = torch.zeros(5, 3, dtype=torch.long)

# 리스트로부터 직접 생성
x = torch.tensor([5.5, 3])

# 기존 Tensor 기반 생성 (속성 복사)
x = x.new_ones(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.float)
```

---

## 4. Tensor 속성 확인

Tensor는 다양한 속성을 가집니다:

```python
print(x.size())  # 크기 확인
print(x.shape)   # shape로도 가능
print(x.dtype)   # 데이터 타입
print(x.device)  # 저장 위치 (CPU or GPU)
```

---

## 5. Tensor 연산

기본적인 연산 방식:

```python
y = torch.rand(5, 3)
print(x + y)              # 연산 1
print(torch.add(x, y))    # 연산 2

# 결과 Tensor를 특정 변수에 저장
result = torch.empty(5, 3)
torch.add(x, y, out=result)

# In-place 연산 (메모리 재사용)
y.add_(x)
```

---

## 6. Tensor 크기 변경

Tensor는 유연하게 크기를 바꿀 수 있습니다:

```python
x = torch.randn(4, 4)
y = x.view(16)    # 1차원 변경
z = x.view(-1, 8) # -1은 크기 자동 계산
```

---

## 7. NumPy 상호 운용성

Tensor는 NumPy 배열로 쉽게 변환됩니다:

```python
a = torch.ones(5)
b = a.numpy()     # Tensor → ndarray
```

NumPy 배열도 Tensor로 변환 가능합니다:

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)  # ndarray → Tensor
```

---

## 8. CUDA 연산

GPU 사용을 위한 Tensor 전송:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z.to("cpu", torch.double))  # 다시 CPU로 전송
```

---

## 참고 자료

- [PyTorch 공식 튜토리얼 - Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch 공식 문서](https://pytorch.org/docs/)

---

## History  
작성일: `2025-05-24`
