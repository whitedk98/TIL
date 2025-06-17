## 1. 개요 (Overview)

Attention 메커니즘은 Transformer 구조의 핵심으로, 모델이 입력 데이터 내의 **중요한 정보에 선택적으로 집중**할 수 있도록 합니다. 자연어 처리(NLP), 컴퓨터 비전(CV), 추천 시스템 등 다양한 분야에서 사용되며, 특히 **문맥 이해와 관계 분석**에 강점을 가집니다.

---

## 2. Attention이란?

- Attention은 **입력의 모든 요소 간의 관계를 동적으로 평가**하여, 중요한 정보에 높은 가중치를 부여하는 메커니즘입니다.
- 사람의 주의(attention)처럼, 모델이 문장의 특정 단어 또는 이미지의 특정 부분에 집중하도록 학습합니다.
- 입력 전체에서 정보를 고려하기 때문에, **길이와 상관없이 전역 문맥(global context)을 반영**할 수 있습니다.

---

## 3. Query, Key, Value의 정의

Transformer에서 Attention은 세 가지 구성 요소를 기반으로 작동합니다:

| 구성 요소 | 의미 | 비유 |
| --- | --- | --- |
| **Query (Q)** | 모델이 관심을 갖고 있는 기준점 | 질문하는 사람 |
| **Key (K)** | Query가 어떤 Value를 찾기 위해 참조할 기준 정보 | 도서관의 책 제목 목록 |
| **Value (V)** | 실제로 전달할 정보, 정답 또는 내용 | 책의 본문 내용 |

예시:

> "The cat sat on the mat"에서 "cat"이 Query라면, 전체 단어들이 Key/Value가 되어 연관성을 평가하고 가장 관련 있는 단어들에 주의를 집중하게 됩니다.
> 

---

## 4. Scaled Dot-Product Attention 계산 방식

Attention score는 다음 수식으로 계산됩니다:

```
mathematica
복사편집
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

```

- `QKᵀ`: Query와 Key 간 유사도를 내적(dot product)으로 측정합니다.
- `√dₖ`: 차원 수로 나누어 스케일을 조정함으로써 gradient가 지나치게 커지는 것을 방지합니다.
- `softmax`: 유사도를 확률(가중치)로 변환
- `× V`: 확률 기반 가중합으로 Value들을 조합

이 연산을 통해 모델은 현재 Query가 어떤 Value(정보)를 얼마나 중요하게 반영해야 할지 계산하게 됩니다.

---

## 5. Masked Attention vs 일반 Attention

| 종류 | 설명 | 사용 예 |
| --- | --- | --- |
| 일반 Attention | 모든 위치의 Key를 참조 가능 | 인코더 내 Attention |
| **Masked Attention** | 미래 정보(Masked 위치)를 참조하지 못하도록 마스킹 적용 | 디코더에서 다음 토큰 예측 시 사용 (언어 생성 모델) |

---

## 6. Attention의 시각적 이해

Attention의 결과는 행렬 형태(2차원 배열)로 나타나며, 이를 통해 입력 토큰들 간의 연관도(가중치)를 시각적으로 분석할 수 있습니다. 이 행렬을 **Attention Map** 또는 **Attention Heatmap**이라고 하며, 모델이 문장에서 어떤 단어에 더 주목했는지를 보여줍니다.

### ▸ Attention Score 행렬

```
bash
복사편집
      the  cat  sat  on  the  mat
the   0.1  0.3  0.2  0.1  0.1  0.2
cat   0.2  0.4  0.2  0.1  0.0  0.1
sat   0.1  0.2  0.3  0.2  0.1  0.1
...

```

### ▸ 구성 설명

- 이 행렬은 `행(Row)`을 기준으로 해석합니다. 즉, 각 행은 **Query 토큰**에 해당합니다.
- 각 **Column**은 해당 Query가 주목한 **Key**에 대한 가중치입니다.
- 각 값은 해당 Query가 해당 Key에 얼마나 집중했는지를 나타내는 **가중치(softmax score)**입니다.
- 각 행의 합산은 softmax로 처리된 결과에 의해 항상 1입니다.