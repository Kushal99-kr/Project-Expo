# ANN
import numpy as np
import matplotlib.pyplot as plt
def plot_sigmoid():
x=np.linspace(-10,10)
y=1/(1+np.exp(-x))
plt.plot(x,y)
plt.xlabel('input')
plt.ylabel('sigmoid output')
plt.title('sigmoid activation function')
plt.grid('true')
plt.show()
plot_sigmoid()

import numpy as np
import matplotlib.pyplot as plt
def plot_threshold():
x=np.linspace(-10,10)
z=np.heaviside(x,1)
plt.plot(x,z)
plt.title("Threashold activation function")
plt.xlabel("x")
plt.ylabel("step(x)")
plt.grid(True)
plt.show()
plot_threshold()

import numpy as np
import matplotlib.pyplot as plt
def plot_piecewise_linear():
x=np.linspace(-2,2)
z=np.piecewise(x,[x>=0.5,(x<0.5)&(x>-0.5),x<=-0.5],[1,lambda x:x ,0])
plt.plot(x,z)
plt.title("piecewise linear activation function")
plt.xlabel("x")
plt.ylabel("piecewise linear(x)")
plt.grid(True)
plt.show()
plot_piecewise_linear()

WEEK 2

class McCullochPittsNeuron:
def __init__(self,weights,threshold):
self.weights=weights
self.threshold=threshold
def activate(self,inputs):
weighted_sum=0
for i in range(len(self.weights)):
weighted_sum+=self.weights[i]*inputs[i]
if weighted_sum>=self.threshold:
return 1
else:
return 0
def AND_gate(x1,x2):
weights=[1,1]
threshold=2
and_neuron=McCullochPittsNeuron(weights,threshold)
return and_neuron.activate([x1,x2])

print("And gate")
print("0 AND 0=",AND_gate(0,0))
print("0 AND 1=",AND_gate(0,1))
print("1 AND 0=",AND_gate(1,0))
print("1 AND 1=",AND_gate(1,1))

WEEK 3

class McCullochPittsNeuron:
def __init__(self,weights,threshold):
self.weights=weights
self.threshold=threshold
def activate(self,inputs):
weighted_sum=0
for i in range(len(self.weights)):
weighted_sum+=self.weights[i]*inputs[i]
if weighted_sum>=self.threshold:
return 1
else:
return 0
def NOT_gate(x):
weight=-1
threshold=0
not_neuron=McCullochPittsNeuron([weight],threshold)
return not_neuron.activate([x])
print("NOT gate")
print("NOT 0",NOT_gate(0))
print("NOT 1",NOT_gate(1))

WEEK 4

import numpy as np
class McCullochPittsNeuron:
def __init__(self,weights,threshold):
self.weights=weights
self.threshold=threshold
def activate(self,inputs):
weighted_sum=sum(w*x for w,x in zip(self.weights,inputs))
if weighted_sum>=self.threshold:
return 1
else:
return 0
def xor(x1,x2):
return 1 if (x1!=x2) else 0
class XOR_MultiLayer_MP:
def __init__(self):
self.hidden_layer1=[
McCullochPittsNeuron(weights=[-10,10],threshold=10),
McCullochPittsNeuron(weights=[10,-10],threshold=10)
]
self.hidden_layer2=[
McCullochPittsNeuron(weights=[10,10],threshold=10)
]
self.output_neuron=McCullochPittsNeuron(weights=[10,10],threshold=10)
def forward(self,x1,x2):
hidden_layer1_output=[neuron.activate([x1,x2]) for neuron in self.hidden_layer1]
hidden_layer2_output=[neuron.activate(hidden_layer1_output) for neuron in self.hidden_layer2]
output=self.output_neuron.activate(hidden_layer2_output)
return output

ANN Lab week-5 Program

''''week-5: simulate perceptron learning network and separate the boundaries.plot the points assumed in the respective quadrants using different symbols for identification.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

def generate_random_points(num_points):
np.random.seed(0)
return np.random.rand(num_points, 4) * 10 - 5

def assign_labels(points):
return np.array([1 if point[0] * point[1] >= 0 else 0 for point in points])

def plot_points(points, labels, clf):
plt.figure(figsize=(8, 6))

for i in range(len(points)):
if labels[i] == 1:
plt.scatter(points[i, 0], points[i, 1], color='blue', marker='o')
else:
plt.scatter(points[i, 0], points[i, 1], color='red', marker='x')

def plot_decision_boundary(points, labels, clf):
coef = clf.coef_[0]
intercept = clf.intercept_[0]
x_values = np.linspace(-5, 5)
y_values = -(coef[0] * x_values + intercept) /coef[1]
plt.plot(x_values, y_values, color='black')

plt.title('Perceptron Decision Boundary')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

num_points = 10
points = generate_random_points(num_points)
labels = assign_labels(points)
print(points)


clf = Perceptron(max_iter=1000, tol=1e-3)
clf.fit(points, labels)

plot_points(points, labels, clf)
plot_decision_boundary(points, labels, clf)

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import metrics
names=['sepallength','sepalwidth','petallength','petalwidth','class']
dataset=pd.read_csv("iris.csv",names=names)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"classification report:\n {metrics.classification_report(y_test, y_pred)}")
