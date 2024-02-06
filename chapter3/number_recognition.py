import pickle
import sys, os
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist

sys.path.append(os.pardir)


def initialize_mnist():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_train, t_train, x_test, t_test


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def mnist_show():
    x_train, t_train, x_test, t_test = initialize_mnist()
    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)


def get_data():
    x_train, t_train, x_test, t_test = initialize_mnist()
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a)
    return np.exp(a - c) / np.sum(exp_a)


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


def run_neuralnet():
    x, t = get_data()
    network = init_network()
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    # 정답률
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


run_neuralnet()