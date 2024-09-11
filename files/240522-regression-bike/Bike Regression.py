import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

macbook = pd.read_csv('~/aiffel/workplace/240522-bike-regression/data/macbook.csv')
print(macbook.shape)
macbook.head()

# 실행한 브라우저에서 바로 그림을 볼 수 있게 해줌
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' # 더 높은 해상도로 출력한다.

plt.scatter(macbook['used_years'], macbook['price'])
plt.show()

# np.corrcoef(x, y)를 사용합니다.
np.corrcoef(macbook['used_years'], macbook['price'])

x = macbook["used_years"].values
y = macbook["price"].values

def model(x, w, b):
    y = w * x + b
    return y

# x축, y축 그리기
plt.axvline(x=0, c='black')
plt.axhline(y=0, c='black')

# y = wx + b 일차함수 그리기
x = np.linspace(0, 8, 9)
y = model(x, w=-20, b=140) # y = -20x + 140
plt.plot(y)

# 나의 (x, y) 점 찍기
x_data = [2, 5, 6]
y_data = [100, 40, 20]
plt.scatter(x_data, y_data, c='r', s=50)

plt.show()

w = 3.1
b = 2.3

x = np.linspace(0, 5, 6)
y = model(x, w, b) # y = 3.1x + 2.3
plt.plot(y, c='r')

plt.scatter(macbook['used_years'], macbook['price'])
plt.show()

x = macbook["used_years"].values
x

prediction = model(x, w, b) # 현재 w = 3.1, b = 2.3
prediction

macbook['prediction'] = prediction
macbook.head()

macbook['error'] = macbook['price'] - macbook['prediction']
macbook.head()

def RMSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    rmse = mse ** 0.5        # MSE의 제곱근
    return rmse

x = macbook["used_years"].values
y = macbook["price"].values

predictions = model(x, w, b)
print(predictions)


rmse = RMSE(predictions, y)
rmse

def loss(x, w, b, y):
    predictions = model(x, w, b)
    L = RMSE(predictions, y)
    return L

def gradient(x, w, b, y):
    dw = (loss(x, w + 0.0001, b, y) - loss(x, w, b, y)) / 0.0001
    db = (loss(x, w, b + 0.0001, y) - loss(x, w, b, y)) / 0.0001
    return dw, db

LEARNING_RATE = 1

x = macbook["used_years"].values
y = macbook["price"].values

w = 3.1
b = 2.3
w, b

losses = []

for i in range(1, 2001):
    dw, db = gradient(x, w, b, y)   # 3, 4번: 모델이 prediction을 예측하고, 손실함수값을 계산함과 동시에 기울기 계산
    w -= LEARNING_RATE * dw         # 5번: w = w - η * dw 로 업데이트
    b -= LEARNING_RATE * db         # 5번: b = b - η * db 로 업데이트
    L = loss(x, w, b, y)            # 현재의 loss 값 계산
    losses.append(L)                # loss 값 기록
    if i % 100 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))


plt.plot(losses)
plt.show()

# 모델에 넣을 x 값들 준비
x = np.linspace(0, 5, 6)

# x, w, b를 모델에 넣어 y값 출력
y = model(x, w, b)

# 일차함수 y 그리기
plt.plot(y, c="r")


# 원본 데이터 점찍기
plt.scatter(macbook['used_years'], macbook['price'])
plt.show()

test = pd.read_csv("~/aiffel/workplace/240522-bike-regression/data/macbook_test.csv")
print(test.shape)
test.head()

test_x = test['used_years'].values
test_y = test['price'].values

prediction = model(test_x, w, b)
test['prediction'] = prediction
test

test['error'] = test['price'] - test['prediction']
test

rmse = ((test['error'] ** 2).sum() / len(test)) ** 0.5
rmse

# 모델 일차함수 그리기
x = np.linspace(0, 5, 6)
y = model(x, w, b)
plt.plot(y, c="r")

# 실제 데이터 값
plt.scatter(test['used_years'], test['price'])

# 모델이 예측한 값
plt.scatter(test['used_years'], test['prediction'])
plt.show()