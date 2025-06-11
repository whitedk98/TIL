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

- 손실 함수: `CrossEntropyLoss`
- 옵티마이저: `AdamW`
- 학습 Epoch: 100+
- 학습률: `4e-4`

```python
loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
```

- 에폭별 Loss 시각화도 진행

---

## 4. 성능 및 테스트 결과

### ▸ 예측 방식

- Autoregressive 예측 (한 토큰씩 반복 생성)
- `generate()` 함수로 반복 샘플링
- `top_k`, `temperature`, `eos_id` 조절 가능

### ▸ 예시 입력: `"Dobby is"`

```
"Dobby is a house-elf.… said Harry."
"Dobby is had to punish himself, sir…”"
"Dobby is used to death threats …"
```

## History
- 작성일: 2025-06-09