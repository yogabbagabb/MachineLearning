import numpy as np
from sklearn.linear_model import LogisticRegression as expit

def thetaFunc(y, theta, x):

    deg = 6

    spot = 0
    sum = theta[spot]

    spot += 1
    for i in range(1, deg + 1):
        for j in range(i + 1):
            sum += theta[spot] * x**(i - j) * y**(j)
            spot += 1
    return -sum


def constructVariations(X, deg):

    features = np.zeros((len(X), 27)) 
    spot = 0

    for i in range(1, deg + 1):
        for j in range(i + 1):

            features[:, spot] = X[:,0]**(i - j) * X[:,1]**(j)
            spot += 1

    return features

if __name__ == '__main__':
    data = np.loadtxt("ex2data2.txt", delimiter = ",")
    X,Y = np.split(data, [len(data[0,:]) - 1], 1)

    X = constructVariations(X, 6)
    rawX = np.copy(X)

    oneArray = np.ones((len(X),1))
    X = np.hstack((oneArray, X))
    trial = expit(solver = 'sag')
    trial = trial.fit(X = X,y = np.ravel(Y))

    from matplotlib import pyplot as plt

    theta = trial.coef_.ravel()

    x = np.linspace(-1, 1.5, 100)
    y = np.linspace(-1,1.5,100)
    z = np.empty((100,100))

    print(trial.coef_)
    print(trial.intercept_)
    
    xx,yy = np.meshgrid(x,y)
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = thetaFunc(yy[i][j], theta, xx[i][j])
    z -= trial.intercept_

    plt.contour(xx,yy,z > 1, alpha=0.8)
    plt.scatter(rawX[:, 0], rawX[:, 1], c=Y, linewidth=0, s=50)
    plt.show()
