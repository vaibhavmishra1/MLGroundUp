from nn import NeuralNetwork
from engine import Value
import random

nn = NeuralNetwork(4, 1)
x = [[Value(random.uniform(-1, 1)) for _ in range(4)] for _ in range(128)]
y = [Value(random.uniform(-1, 1)) for _ in range(128)]
for i in range(1000):
    loss = Value(0)
    for j in range(128):
        output = nn(x[j])
        loss += (output[0] - y[j]) ** 2
        # if j==0:
        #     print("x[0] = ", x[j], "y[0] = ", y[j])
        #     print("output[0] = ", output[0])
    loss.backward()
    for param in nn.parameters():
        param.data -= 0.001 * param._grad
    print(loss.data)
    loss.zero_grad()
    


