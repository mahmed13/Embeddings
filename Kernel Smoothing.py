import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n=100

X = np.random.uniform(low=-5, high=5, size=(n,))
noise = np.random.normal(0, 0.1, n) # epsilon
y = np.sin(X)+noise

def kernel(a):
    return 1.0/(math.sqrt(2*math.pi)) * math.exp((-math.pow(a,2)/2))

# Nadaraya-Watson estimator
def estimator(x_k,X,y,h):
    num = 0
    denom = 0
    for x_i,y_i in zip(X,y):
        num += y_i*kernel(abs(x_i-x_k)/h)
        denom += kernel(abs(x_i-x_k)/h)
    return num/denom

def loss_func(y,y_hat):
    return (y-y_hat)**2

n = 200
X = np.random.uniform(low=-5, high=5, size=(n,))
noise = np.random.normal(0, 0.1, n) # epsilon
y = np.sin(X)+noise
plt.scatter(X,y)

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

H = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
for h in H:
    f_hat = [estimator(x_k,X_train,y_train,h) for x_k in X_train]

    print('Train error:',np.mean(loss_func(y_train, f_hat)))



# plt.scatter(X, np.sin(X), alpha=0.5)
# plt.plot(X,y)
# plt.title('Scatter plot -- 2pts')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()

