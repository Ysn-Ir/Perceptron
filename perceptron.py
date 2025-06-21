import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def ramp(x):
    return np.where(x>0,1,0)
class Perceptron:
    def __init__(self,lr,iter):
        self.lr=lr
        self.iter=iter
        self.activation=ramp
        self.weight=None
        self.bias=None
    def fit(self,data):
        x=data[0]
        y=data[1]
        n,m=x.shape
        self.weight=np.zeros(m)
        self.bias=0
        for i in range(self.iter):
            for id,xi in enumerate(x):
                z=np.dot(self.weight,xi)+self.bias
                y_pred=self.activation(z)
                update=(y[id]-y_pred)
                self.weight+=self.lr*(update*xi)
                self.bias+=self.lr*(update)
    def predict(self,x):
        z=np.dot(x,self.weight)+self.bias
        return self.activation(z)
    


def generate_data(n_samples=100, random_state=42):
    np.random.seed(random_state)
    
    # Generate class 0 points
    x0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    y0 = np.zeros((n_samples // 2,))
    
    # Generate class 1 points
    x1 = np.random.randn(n_samples // 2, 2) + np.array([6, 6])
    y1 = np.ones((n_samples // 2,))
    
    # Combine
    X = np.vstack((x0, x1))
    Y = np.concatenate((y0, y1))
    
    return X, Y
X, Y = generate_data(2000)


p = Perceptron(lr=0.01, iter=2000)
p.fit((X, Y))

# Predict and compute accuracy
y_pred = p.predict(X)
accuracy = np.mean(y_pred == Y)
print(f"Accuracy: {accuracy * 100:.2f}%")


def plot_decision_boundary(perceptron, X, Y):
    w = perceptron.weight
    b = perceptron.bias
    
    A = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    
    # Compute slope
    if w[1] != 0:
        B = -(w[0] * A + b) / w[1]
    else:
        B = np.full_like(A, -b)  # vertical line
    
    # Plot the data
    plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], color='red', label='Class 0')
    plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], color='blue', label='Class 1')
    
    # Plot the decision line
    plt.plot(A, B, color='black', linestyle='--', label='Decision boundary')
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.grid(True)
    plt.show()

plot_decision_boundary(p,X,Y)
        
