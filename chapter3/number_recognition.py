import sys, os
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist

sys.path.append(os.pardir)


def initialize_mnist():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
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
    img = img.reshape(28,28)
    print(img.shape)

    img_show(img)

mnist_show()
