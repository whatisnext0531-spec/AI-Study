+++
description = '대용량 네트워크 트래픽 데이터를 활용한 다중 분류 경진대회. 첫 제출과 최종 제출 코드를 비교하며 Feature Engineering, 5-Fold 검증, LightGBM·Random Forest 앙상블을 통한 개선 과정 정리'
draft = false
+++

# 네트워크 침입 탐지 머신러닝 분류 미니 경진대회

## 프로젝트 개요

네트워크 패킷 트래픽 데이터를 이용하여 정상 트래픽과 여러 유형의 네트워크 공격을 분류하는 머신러닝 미니 경진대회

경진대회 시작 전 머신러닝 분류 문제 해결을 위한 전체 작업 순서를 정리하고, 해당 순서에 따라 데이터 확인 → 전처리 → 모델링 → 제출 과정 진행

진행 중 모든 코드 변경 사항을 기록하지는 못했기 때문에 경진대회 종료 후 **첫 번째 제출 코드와 마지막 제출 코드를 비교하고, 성능 향상에 영향을 주었을 가능성이 있는 변화들을 AI를 활용하여 분석**

---

## 1. 문제 정의

### 예측 목표

네트워크 패킷 트래픽을 분석하여 각 트래픽이 정상 상태인지, 또는 어떤 유형의 네트워크 공격인지 예측하는 **다중 분류 문제**

Target인 `Label`은 다음 5개 클래스로 구성

- Benign
- DDOS
- DoS
- Brute Force
- Infiltration

### 데이터 규모

- Train: **4,638,804행 × 43열**
- Test: **1,159,701행 × 43열**

Train 데이터만 약 463만 건에 해당하는 대용량 데이터

→ 모델의 예측 성능뿐만 아니라 학습 시간과 메모리 사용량도 함께 고려할 필요

### 클래스 불균형

EDA를 통해 Target의 클래스 비율 확인

| Label | 비율 |
|---|---:|
| Benign | 약 68.81% |
| DDOS | 약 11.86% |
| DoS | 약 11.17% |
| Brute Force | 약 6.58% |
| Infiltration | 약 1.59% |

정상 트래픽인 Benign이 약 69%를 차지한 반면 Infiltration은 약 1.6%에 불과한 불균형 데이터

→ Accuracy만 확인하기보다 상대적으로 적은 공격 클래스의 분류 성능도 함께 고려할 필요

최종 코드에서는 Benign을 제외한 공격 클래스의 F1-score를 `weighted` 방식으로 계산하는 별도의 평가 함수 사용

---

## 2. 분석 순서

모델을 바로 학습하기보다 전체 작업 순서를 먼저 정리

1. 문제 정의 및 Target·평가지표 확인
2. 데이터 로드 및 기본 구조 확인
3. EDA와 클래스 불균형 확인
4. Train / Validation 분할
5. 결측치·범주형 변수 등 데이터 전처리
6. Feature Engineering
7. 모델 학습 및 검증
8. 앙상블과 최종 Test 예측
9. 제출 파일 확인

실제 분석 과정에서는 데이터의 특성과 모델 결과에 따라 일부 과정 추가 및 변경

---

## 3. 기본 전처리

### unique_id 제거

`unique_id`는 각 행을 구별하기 위한 식별자로 문자와 숫자가 섞인 문자열 형태

예측에 사용할 Feature라기보다 제출 시 각 행을 식별하기 위해 필요한 값으로 판단

→ 제출용 ID로 별도 보관한 뒤 Train과 Test에서는 제거

```python
train_id = train['unique_id']
test_id = test['unique_id']

train = train.drop(columns=['unique_id'], errors='ignore')
test = test.drop(columns=['unique_id'], errors='ignore')
```

데이터에 존재할 수 있는 무한대 값은 이후 처리를 위해 결측치로 변환

```python
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)
```

---

## 4. Label 오타 처리

원본 데이터에 `Infilteration`이라는 Label 존재

학습 과정에서는 이를 올바른 표기인 `Infiltration`으로 수정

```python
train['Label'] = train['Label'].replace({
    'Infilteration': 'Infiltration'
})
```

처음에는 단순한 오타로 판단하여 수정

그러나 이후 **학습 과정에서 사용하는 Label과 실제 제출 시스템이 요구하는 Label 형식은 다를 수 있다는 점 확인**

→ 최종 제출 과정에서 별도의 Label 복원 과정 추가

---

## 5. 첫 번째 제출

첫 번째 제출에서는 비교적 단순한 구조의 모델링 진행

### 데이터 분할

Train 데이터를 8:2로 분할하고 클래스 비율 유지를 위해 `stratify` 적용

```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
```

→ 클래스 불균형을 고려한 Train / Validation 분할

### Timestamp 가공

첫 번째 제출에서는 Timestamp에서 두 가지 시간 정보 추출

```python
combined['Hour'] = combined['Timestamp'].dt.hour
combined['DayOfWeek'] = combined['Timestamp'].dt.dayofweek
```

- `Hour`: 트래픽 발생 시간대
- `DayOfWeek`: 트래픽 발생 요일

### Scaling

수치형 변수에 `RobustScaler` 적용

```python
scaler = RobustScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
test[num_cols] = scaler.transform(test[num_cols])
```

### LightGBM 단일 모델

첫 번째 제출 모델로 LightGBM 사용

```python
model = LGBMClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```

클래스 불균형을 고려한 `class_weight='balanced'` 적용

→ 이후 개선 과정의 기준 모델로 활용

---

## 6. Feature Engineering

최종 코드에서는 기존 Feature를 그대로 사용하는 것에서 나아가 네트워크 트래픽의 특성을 새로운 변수로 표현

### Pkt_Ratio

Forward와 Backward 방향으로 전달된 패킷 수의 비율

```python
df['Pkt_Ratio'] = (
    df['Tot Fwd Pkts'] /
    (df['Tot Bwd Pkts'] + 1e-5)
).astype('float32')
```

`Tot Fwd Pkts / Tot Bwd Pkts`

→ 패킷이 양방향으로 얼마나 불균형하게 오갔는지를 하나의 변수로 표현

→ 정상 통신과 공격 트래픽의 통신 방향 패턴 차이를 반영하기 위한 파생변수

### Byte_Ratio

Forward와 Backward 방향으로 전송된 데이터량의 비율

```python
df['Byte_Ratio'] = (
    df['TotLen Fwd Pkts'] /
    (df['TotLen Bwd Pkts'] + 1e-5)
).astype('float32')
```

`TotLen Fwd Pkts / TotLen Bwd Pkts`

→ 통신 과정에서 어느 방향으로 데이터가 더 집중되어 있는지 표현

### Byte_per_Pkt

패킷 하나당 평균적인 데이터량

```python
df['Byte_per_Pkt'] = (
    df['Flow Byts/s'] /
    (df['Flow Pkts/s'] + 1e-5)
).astype('float32')
```

`Flow Byts/s / Flow Pkts/s`

→ 기존의 초당 Byte와 초당 Packet 정보를 결합하여 패킷 하나당 데이터량을 새로운 Feature로 표현

### Dst_Port_Freq

목적지 Port가 Train 데이터에서 얼마나 자주 등장하는지를 나타내는 빈도 Feature 추가

```python
port_counts = X['Dst Port'].value_counts()

X['Dst_Port_Freq'] = X['Dst Port'].map(port_counts).fillna(0).astype('int32')
test['Dst_Port_Freq'] = test['Dst Port'].map(port_counts).fillna(0).astype('int32')
```

Port 값 자체뿐만 아니라 **특정 Port가 데이터에서 얼마나 자주 사용되는지에 대한 정보 추가**

---

## 7. Timestamp Feature 확장

첫 제출에서는 `Hour`, `DayOfWeek`만 사용

최종 코드에서는 Timestamp를 보다 세밀하게 분해

```python
combined['Month'] = combined['Timestamp'].dt.month
combined['Day'] = combined['Timestamp'].dt.day
combined['Hour'] = combined['Timestamp'].dt.hour
combined['Minute'] = combined['Timestamp'].dt.minute
combined['Second'] = combined['Timestamp'].dt.second
combined['DayOfWeek'] = combined['Timestamp'].dt.dayofweek
```

추가로 Timestamp 전체를 초 단위의 연속적인 숫자로 표현

```python
combined['Timestamp_epoch'] = (
    combined['Timestamp'].astype('int64') // 10**9
)
```

→ 첫 번째 제출보다 시간 정보를 다양한 형태로 모델에 제공

---

## 8. 전처리 전략 변경

첫 제출에서 사용했던 일부 전처리를 최종 코드에서는 제거

### 고상관 변수 삭제 → 유지

첫 번째 코드에서는 일부 상관관계가 높은 Feature 제거

최종 코드에서는 해당 Feature를 제거하지 않고 유지

→ 트리 기반 모델에서는 Feature 간 상관관계가 높다는 이유만으로 반드시 제거할 필요가 없다는 점을 고려한 전략 변경

### RobustScaler 사용 → 제거

첫 번째 제출에서는 수치형 변수에 `RobustScaler` 적용

최종 학습 과정에서는 Scaling 제거

LightGBM과 Random Forest 모두 결정트리 기반 모델로 Feature Scaling이 필수적이지 않은 모델

→ 일괄적인 전처리 적용보다 **모델의 특성에 필요한 전처리를 선택하는 방향으로 변경**

---

## 9. 검증 방식 개선

### 첫 번째 제출

한 번의 Train / Validation 분할

`Stratified 8:2 Split`

### 최종 제출

`StratifiedKFold` 기반 5-Fold 교차검증 적용

```python
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

각 Fold에서 원본 데이터의 클래스 비율 유지

→ 전체 데이터를 여러 번 Train과 Validation으로 활용

→ 한 번의 Validation 결과에만 의존하지 않고 여러 데이터 분할에서 모델 성능 확인

---

## 10. LightGBM + Random Forest 앙상블

첫 제출에서는 LightGBM 단일 모델 사용

최종 코드에서는 Random Forest 추가

```python
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42 + fold,
    n_jobs=-1
)
```

LightGBM과 Random Forest의 예측 확률을 결합하는 **Soft Voting 방식 적용**

```python
val_proba_ensemble = (
    val_proba_lgb * 0.6
    + val_proba_rf * 0.4
)
```

### 앙상블 비율

**LightGBM 60% + Random Forest 40%**

LightGBM을 주력 모델로 사용하면서 서로 다른 학습 방식을 가진 Random Forest의 예측 결과를 함께 반영

Test 데이터 역시 각 Fold에서 생성된 예측 확률을 결합한 뒤 평균하여 최종 예측값 생성

---

## 11. 제출 단계에서 발견한 Label 문제

이번 경진대회에서 모델 외적으로 가장 기억에 남았던 문제

원본 데이터의 Label

`Infilteration`

학습 과정에서 수정한 Label

`Infiltration`

최종 코드에서는 제출 직전에 다시 원래 Label로 복원

```python
test_preds_labels_fixed = [
    'Infilteration' if label == 'Infiltration' else label
    for label in test_preds_labels
]
```

처리 과정

```text
원본 데이터
Infilteration
      ↓
학습 단계
Infiltration
      ↓
제출 단계
Infilteration
```

→ 모델 학습에 사용하는 Label뿐만 아니라 **제출 파일의 Label과 형식이 채점 시스템의 기준과 일치하는지 확인할 필요**

→ 모델링 이후의 제출 과정 역시 전체 머신러닝 파이프라인의 일부라는 점 확인

---

## 12. 첫 제출과 마지막 제출 비교

| 항목 | 첫 번째 제출 | 마지막 제출 |
|---|---|---|
| 모델 | LightGBM | LightGBM + Random Forest |
| 검증 | Stratified 8:2 Split | Stratified 5-Fold |
| LightGBM Tree 수 | 300 | 800 |
| Random Forest | 사용하지 않음 | 300 Trees |
| 앙상블 | 없음 | Soft Voting |
| 앙상블 비율 | - | LGBM 0.6 / RF 0.4 |
| Pkt_Ratio | 없음 | 추가 |
| Byte_Ratio | 없음 | 추가 |
| Byte_per_Pkt | 없음 | 추가 |
| Dst_Port_Freq | 없음 | 추가 |
| Timestamp | Hour, DayOfWeek | 시간 Feature 세분화 + Epoch |
| 일부 고상관 변수 | 삭제 | 유지 |
| RobustScaler | 사용 | 제거 |
| 제출 Label 복원 | 없음 | 적용 |

첫 번째 제출 → 마지막 제출 과정에서 단순한 하이퍼파라미터 수정이 아닌 **Feature Engineering, 검증 방식, 모델 구성, 제출 과정까지 전체 파이프라인의 변화**

---

## 13. 점수 향상 요인 분석

경진대회 중 약 **0.94 부근에서 점수 정체**

이후 첫 번째 제출과 마지막 제출 사이에 적용된 주요 변화

- 비율 기반 파생변수 추가
- Dst Port 빈도 Feature 추가
- Timestamp Feature 확장
- Stratified 5-Fold 교차검증 적용
- LightGBM + Random Forest 앙상블
- 전처리 방식 변경
- 제출 Label 처리 수정

그러나 각 변경 사항을 하나씩 적용하면서 모든 점수를 기록하지 못한 상황

→ 개별 변경 사항이 최종 점수 상승에 얼마나 기여했는지 정확한 분리 불가

→ `Pkt_Ratio 추가로 몇 점 상승`, `Random Forest 추가로 몇 점 상승`과 같은 인과관계 판단 불가

→ 여러 변경이 함께 적용되면서 최종 성능이 개선되었을 가능성

**가장 큰 아쉬움: 실험별 변경 사항과 점수를 체계적으로 기록하지 못한 점**

---

## 14. 프로젝트를 통해 배운 점

### 전체 파이프라인의 중요성

처음에는 모델 성능 향상에 집중했지만 실제 작업을 통해 확인한 전체 과정

**데이터 → 전처리 → Feature → 모델 → 검증 → 예측 → 제출**

Label 문제를 통해 모델 학습 이후의 제출 파일까지 하나의 파이프라인으로 확인해야 한다는 점 경험

### Feature Engineering의 의미

기존 Feature를 그대로 사용하는 것과 기존 변수의 관계를 새로운 Feature로 표현하는 것의 차이 확인

`Pkt_Ratio`, `Byte_Ratio`, `Byte_per_Pkt` 생성 과정에서 단순한 변수 추가보다 **데이터의 의미를 모델이 활용할 수 있는 형태로 표현하는 과정의 중요성 확인**

### 모델에 맞는 전처리 선택

첫 제출에서는 Scaling과 일부 고상관 변수 제거 적용

최종 코드에서는 일부 과정 제거

→ 일반적인 전처리 방법을 일괄적으로 적용하기보다 **데이터와 모델의 특성을 고려한 선택의 필요성 확인**

### 검증 방식의 중요성

단일 Train / Validation 분할에서 Stratified 5-Fold로 변경

→ 모델 자체뿐만 아니라 **모델의 성능을 어떤 방식으로 검증할 것인지도 중요한 모델링 과정**

---

## 15. 아쉬웠던 점과 다음 목표

### 아쉬웠던 점

실험 과정에 대한 체계적인 기록 부족

코드를 수정하면서 제출 점수를 확인했지만 다음 정보들을 매번 기록하지 못함

- 변경한 코드
- 변경 이유
- 변경 전 Validation 점수
- 변경 후 Validation 점수
- 실제 제출 점수

→ 첫 번째 제출과 마지막 제출의 비교는 가능하지만 중간 단계의 개별 변화가 성능에 미친 영향 분석에는 한계

### 다음 경진대회에서 개선할 점

실험별 변경 사항과 결과 기록

| 실험 | 변경 사항 | CV / Validation | 제출 점수 | 판단 |
|---|---|---:|---:|---|
| 01 | Baseline | - | - | 기준 |
| 02 | 파생변수 추가 | - | - | 비교 |
| 03 | Timestamp 확장 | - | - | 비교 |
| 04 | 5-Fold 적용 | - | - | 비교 |
| 05 | Random Forest 추가 | - | - | 비교 |
| 06 | Soft Voting | - | - | 비교 |
| 07 | Label 처리 수정 | - | - | 최종 |

다음 프로젝트의 목표

**가설 설정 → 변경 → 실험 → 결과 → 판단**

단순히 가장 높은 점수의 최종 코드만 남기는 것이 아니라 **성능 개선 과정과 판단 근거까지 기록하는 프로젝트 진행**