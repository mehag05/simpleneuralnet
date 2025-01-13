import matplotlib.pyplot as plt
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, data):
        features = [sample[0] for sample in data]
        labels = [sample[1] for sample in data]

        self.input = np.array(features)

        self.w0 = np.random.rand(3, 2)
        self.w1 = np.random.rand(4, 3)
        self.w2 = np.random.rand(2, 4)
        self.w3 = np.random.rand(1, 2)
        self.biases = np.random.rand(4)
        self.a_1 = np.zeros((len(features), 3, 1))
        self.a_2 = np.zeros((len(features), 4, 1))
        self.a_3 = np.zeros((len(features), 2, 1))
        self.a_4 = np.zeros((len(features), 1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, input):
        input = np.array(input)
        z1 = self.w0 @ input + self.biases[0]
        self.a_1 = self.sigmoid(z1)
        z2 = self.w1 @ self.a_1 + self.biases[1]
        self.a_2 = self.sigmoid(z2)
        z3 = self.w2 @ self.a_2 + self.biases[2]
        self.a_3 = self.sigmoid(z3)
        z4 = self.w3 @ self.a_3 + self.biases[3]
        self.a_4 = self.sigmoid(z4)
        return self.a_4

    def crossentropy_loss(self, y_i, y_hat):
        y_i = np.array(y_i)
        return -np.sum(y_hat * np.log2(y_hat) + (1 - y_i) * np.log2(1 - y_hat))
    
    def backprop(self, y_i, input):
        # m is a list of (input, result) pairs where input is a list of features and result is a binary value
        cost_times_m = 0
        y_hat = self.feedforward(input)
        cost_times_m += self.crossentropy_loss(y_i, y_hat)

        cost = 1/len(y_i) * cost_times_m

        y_hat = np.array(self.a_4)
        y_i = np.array([y_i]).reshape(y_hat.shape)

        delta_4 = (y_hat - y_i) * (self.a_4 * (1 - self.a_4))
        self.d_C_d_W3 = delta_4 @ self.a_3.T
        self.d_C_d_b3 = delta_4

        delta_3 = (self.w3.T @ delta_4) * (self.a_3 * (1 - self.a_3))
        self.d_C_d_W2 = delta_3 @ self.a_2.T
        self.d_C_d_b2 = delta_3

        delta_2 = (self.w2.T @ delta_3) * (self.a_2 * (1 - self.a_2))
        self.d_C_d_W1 = delta_2 @ self.a_1.T
        self.d_C_d_b1 = delta_2

        delta_1 = (self.w1.T @ delta_2) * (self.a_1 * (1 - self.a_1))
        self.d_C_d_W0 = delta_1 @ np.array(input).T
        self.d_C_d_b0 = delta_1

        return cost

    def gradient_descent(self, alpha):
        self.w0 = self.w0 - alpha * self.d_C_d_W0
        self.w1 = self.w1 - alpha * self.d_C_d_W1
        self.w2 = self.w2 - alpha * self.d_C_d_W2
        self.w3 = self.w3 - alpha * self.d_C_d_W3

        new_biases = [self.biases[0] - alpha * self.d_C_d_b0,
                      self.biases[1] - alpha * self.d_C_d_b1,
                      self.biases[2] - alpha * self.d_C_d_b2,
                      self.biases[3] - alpha * self.d_C_d_b3]
        self.biases = new_biases

def train():
    epochs = 1000
    alpha = 0.01
    cost = []

    # mock data - you can change this to your own data
    data = [
        [[1.0, 2.0], 0],
        [[2.0, 3.0], 0],
        [[3.0, 1.0], 1],
        [[4.0, 5.0], 1],
        [[5.0, 4.0], 1],
        [[6.0, 7.0], 0],
        [[7.0, 6.0], 0],
        [[8.0, 9.0], 1],
        [[9.0, 8.0], 1],
        [[10.0, 11.0], 0]
    ]

    y_i = [sample[1] for sample in data]
    input_features = [ [sample[0][0] for sample in data], [sample[0][1] for sample in data] ]

    SNN = SimpleNeuralNetwork(data)
    for i in range(epochs):
        cost.append(SNN.backprop(y_i, input_features))
        print("cost", cost)
        SNN.gradient_descent(alpha)

    return cost

def plot_cost(cost):
    plt.plot(cost)
    plt.show()


if __name__ == "__main__":
    cost = train()
    plot_cost(cost)







