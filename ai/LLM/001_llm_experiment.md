# LLM Experiment - 1
---


## **1. 환경 구성**

- **OS**: Windows 10 + WSL2
- **Python**: 3.9.20
- **PyTorch**: 2.6 + CUDA 12.6
- **Tokenizer**: `tiktoken` (GPT-2용 BPE 토크나이저)
---

## **2. 구현 및 실습 흐름**

### **▸ Step 1: 텍스트 데이터 전처리**

- 줄바꿈과 공백 제거 후 `cleaned_*.txt`로 저장
- 예시 데이터:
    - Harry Potter
    - Alice in Wonderland

```python
# 전처리 예시
cleaned_text = re.sub(r'\n+', ' ', book_text)
cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
```

---

### ▸ Step 2: 토크나이징

- `tiktoken` 기반 UTF-8 BPE 사용
- 샘플 문장: `"Harry Potter was a wizard."` → 토큰 ID로 인코딩

---

### ▸ Step 3: 데이터 로더 정의

- sliding window 방식으로 input/target 생성
- max_length와 stride 조절 가능

```python
# input: [t0, t1, t2, ..., tn]
# target: [t1, t2, t3, ..., tn+1]
```

---

### ▸ Step 4: Transformer 기반 GPT 모델 정의

- Core 구성
    - MultiHead Attention
    - GELU + FeedForward
    - LayerNorm
    - Causal Masking (`triu`)
- 전체 구조
    - 12층 Transformer Block
    - Embedding → Block → Norm → LM Head

```python
VOCAB_SIZE = 50257
CONTEXT_LENGTH = 128
EMB_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
```

---

### ▸ Step 5: 훈련(Training)
### ▸ 구성

- 손실 함수: `nn.CrossEntropyLoss()`
- 옵티마이저: `torch.optim.AdamW(model.parameters(), lr=4e-4)`
- 학습률 스케줄러는 적용하지 않음 (단일 고정 학습률 사용)
- 배치 사이즈: 64
- Epoch 수: 100
- 평가 주기: 1 epoch 마다 validation loss 측정

```python
for epoch in range(max_epoch):
    for X, Y in train_loader:
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

```

### ▸ 시각화
- 학습이 진행됨에 따라 손실이 점차 감소함을 확인
- 초기 10~20 에폭 동안 손실 감소 폭이 크고, 이후 점진적 수렴

```python
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()
plt.grid(True)
plt.show()
```

## 4. 성능 및 테스트 결과

### ▸ 예측 방식

- Autoregressive 방식으로 토큰 하나씩 예측해 다음 입력으로 재귀 전달
- `generate()` 함수 내에서 temperature, top_k, max_tokens 조절
- 반복 예측 방식 구현:

```python
def generate(model, start_ids, max_tokens=100, temperature=1.0, top_k=50):
    model.eval()
    tokens = start_ids[:]
    for _ in range(max_tokens):
        input_tensor = torch.tensor(tokens[-context_length:]).unsqueeze(0)
        logits = model(input_tensor)[:, -1, :]
        logits = logits / temperature
        top_logits, top_indices = torch.topk(logits, top_k)
        probs = torch.softmax(top_logits, dim=-1)
        next_token = top_indices[0, torch.multinomial(probs, 1)]
        tokens.append(next_token.item())
    return tokens

```

### ▸ 예시 입력: `"Dobby is"`

```
"Dobby is a house-elf.… said Harry."
"Dobby is had to punish himself, sir…”"
"Dobby is used to death threats …"
```

## History
- 작성일: 2025-06-11
- 수정일: 2025-06-12