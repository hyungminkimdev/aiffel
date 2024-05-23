<aside>
🔑 **PRT(Peer Review Template)**

작성자 : 김성연

- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
![img](reason1.png)
- 코드를 막 고쳐도 될지 모르겠네요
    - test data에서 weather 4를 처리해주면 잘 돌아갈 것 같습니다

- [ ]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [ ]  모델 선정 이유
    - [ ]  Metrics 선정 이유
    - [ ]  Loss 선정 이유


- [ ]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [o]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [ ]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [o]  각 실험을 시각화하여 비교하였나요?
    - [ ]  모든 실험 결과가 기록되었나요?

- IQR이 두가지 항목에서 넘어가면 outlier로 처리해서 너무 많은 데이터가 손실되지 않도록 한 점이 인상 깊습니다.

```python
# IQR method
import numpy as np
from collections import Counter

def detect_outliers(data, n, cols):
    outlier_indices = []
    for col in cols:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

Outliers_to_drop = detect_outliers(train, 2, ["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"])
```
- 제가 고민 했던 0을 처리하는 방법에 대한 아이디어를 볼 수 있어서 흥미로웠습니다
```python
from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):
    wind0 = data.loc[data['windspeed'] == 0]
    windnot0 = data.loc[data['windspeed'] != 0]

    # Predict 'windspeed' using weather variables
    col = ['season', 'weather', 'temp', 'humidity', 'atemp', 'day']

    windnot0['windspeed'] = windnot0['windspeed'].astype('str')

    rf = RandomForestClassifier()
    # Fit 'windspeed!=0'
    # model.fit(X_train, Y_train)
    rf.fit(windnot0[col], windnot0['windspeed'])

    # Predict where 'windspeed!=0'
    # model.predict(X_test)
    pred_wind0 = rf.predict(X=wind0[col])

    # Change value of 'wind0' to 'pred_wind0'
    wind0['windspeed'] = pred_wind0

    # Combine 'windnot0' & 'wind0'
    data = windnot0.append(wind0)
    data['windspeed'] = data['windspeed'].astype('float')

    data.reset_index(inplace=True, drop=True)

    return data
```
- 전처리 과정별로 함수로 정리해놓으셔서 보기 편했습니다.


- [o]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [o]  배운 점
    - [o]  아쉬운 점
    - [o]  느낀 점
    - [o]  어려웠던 점
</aside>