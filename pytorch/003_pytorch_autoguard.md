# PyTorch Autograd

PyTorch의 자동 미분 기능인 **Autograd**는 모델 학습 시 역전파(Backpropagation)를 자동으로 수행하게 해주는 핵심 기능입니다.

---

## 1. 개요 (Overview)

**Autograd**는 텐서의 연산 이력을 기록한 **동적 계산 그래프**를 기반으로 자동 미분을 수행합니다.

또한 GPU를 활용하면 대규모 연산을 훨씬 빠르게 처리할 수 있습니다.

---

## 2. Autograd 기본 개념

- `requires_grad=True`로 설정된 Tensor는 연산 시 **기울기 추적**이 활성화됩니다.
- 연산 결과는 **연산 그래프**로 구성되며, `.backward()`를 호출하면 자동으로 미분이 수행됩니다.
- `.grad` 속성으로 기울기를 확인할 수 있습니다.

---

## 3. 기본 예제

```python
python
복사편집
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()
print(x.grad)

```

---

## 4. 역전파(Backpropagation) 원리

```python
python
복사편집
x = torch.randn(3, requires_grad=True)
y = x * 2
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(v)
print(x.grad)

```

---

## 5. 기울기 추적 중지

```python
python
복사편집
x = torch.ones(2, 2, requires_grad=True)
with torch.no_grad():
    y = x * 2
print(y.requires_grad)  # False

```

---

## 6. 기울기 초기화

```python
python
복사편집
model.zero_grad()
optimizer.zero_grad()

```

---

## 7. 실제 사용 예: 선형 회귀 (기본 버전)

```python
python
복사편집
x = torch.randn(10, 3)
y = torch.randn(10, 1)

w = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for _ in range(100):
    pred = x @ w + b
    loss = (pred - y).pow(2).mean()
    loss.backward()

    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
        w.grad.zero_()
        b.grad.zero_()

```

---

## 8. Autograd 주의사항

- `.backward()`는 기본적으로 그래프를 제거함 (`retain_graph=True` 사용 시 보존 가능)
- `.grad`는 누적되므로 `zero_()` 필요
- `with torch.no_grad()` 또는 `torch.no_grad()`는 추론 시 사용

---

## 9. Autograd와 GPU 연산 결합

CUDA를 사용해 GPU에서 Autograd 연산을 수행하려면 Tensor와 모델을 `.to(device)`로 전송해야 합니다.

```python
python
복사편집
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 입력 데이터 및 파라미터를 GPU로 이동
x = torch.randn(100, 3, device=device)
y = torch.randn(100, 1, device=device)

w = torch.randn(3, 1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)

# 학습 루프
for _ in range(100):
    pred = x @ w + b
    loss = (pred - y).pow(2).mean()
    loss.backward()

    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
        w.grad.zero_()
        b.grad.zero_()

# 결과를 CPU로 가져올 수 있음
print(w.cpu().detach())

```

💡 **Tip**: GPU로 연산을 수행할 경우 속도 향상이 매우 크며, 특히 대규모 데이터셋/모델일수록 효과적입니다.

---

## 참고 자료

- PyTorch Autograd 공식 문서
- PyTorch CUDA 공식 문서
- PyTorch 튜토리얼 - Autograd 기초

---

## History

작성일: `2025-06-30`