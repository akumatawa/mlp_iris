import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def flat_theta(theta):
    return np.r_[theta[0].ravel(), theta[1].ravel()]


def initialize_model(input_l, hidden_l, labels):
    epsilon = 0.12
    theta = [np.random.rand(hidden_l, input_l + 1) * 2 * epsilon - epsilon,
             np.random.rand(labels, hidden_l + 1) * 2 * epsilon - epsilon]
    return theta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def forward_propagation(x, theta, m):
    ones_m = np.ones((m, 1))
    activation_vec = np.vectorize(sigmoid)

    # Layers calculation
    a1 = np.c_[ones_m, x]
    a2 = np.c_[ones_m, activation_vec(np.matmul(a1, theta[0].T))]
    a3 = activation_vec(np.matmul(a2, theta[1].T))
    return [a1, a2, a3]


def cost_function(x, desired, theta, lam, m,):
    f_prop = forward_propagation(x, theta, m)

    # y_mat describes classes as rows from eye matrix
    no_classes = np.unique(desired).size
    y_mat = np.empty((m, no_classes))
    eye = np.eye(no_classes)
    for i in range(0, m):
        y_mat[i] = eye[desired[i]]

    ones = np.ones((len(desired), no_classes))

    cost_first = y_mat * np.log(f_prop[2])
    cost_second = (ones - y_mat) * np.log(ones - f_prop[2])

    cost = -np.sum(np.sum(cost_first + cost_second)) / m

    grad = back_propagation(theta, f_prop, y_mat, lam, m)
    return cost, grad


def back_propagation(theta, f_prop, y_mat, lam, m):
    d1 = np.zeros(theta[0].shape)
    d2 = np.zeros(theta[1].shape)
    for i in range(0, m):
        delta_3 = np.array(f_prop[2][i, :] - y_mat[i, :])
        delta_2 = np.matmul(theta[1].T, delta_3.T)[1:].T * \
                  np.vectorize(sigmoid_grad)(np.matmul(f_prop[0][i, :], theta[0].T))

        d1 = d1 + np.matmul(delta_2.reshape(d1.shape[0], 1), f_prop[0][i, :].reshape(d1.shape[1], 1).T)
        d2 = d2 + np.matmul(delta_3.reshape(d2.shape[0], 1), f_prop[1][i, :].reshape(d2.shape[1], 1).T)

    theta_reg = [np.c_[np.zeros((theta[0].shape[0], 1)), theta[0][:, 1:]],
                 np.c_[np.zeros((theta[1].shape[0], 1)), theta[1][:, 1:]]]

    return np.r_[(d1 + lam * theta_reg[0]).ravel() / m,
                 (d2 + lam * theta_reg[1]).ravel() / m]


def predict(theta, x):
    m = x.shape[0]
    h1 = sigmoid(np.matmul(np.c_[np.ones((m, 1)), x], np.transpose(theta[0])).astype(float))
    h2 = sigmoid(np.matmul(np.c_[np.ones((m, 1)), h1], np.transpose(theta[1])).astype(float))
    return np.argmax(h2, axis=1)


def train(x_train, x_test, y_train, y_test, theta, lam, learning_rate, epochs):
    m = x_train.shape[0]
    error = []
    flatten_theta = flat_theta(theta)
    print('Training...')
    for epoch in tqdm(range(epochs)):
        cost, grad = cost_function(x_train, y_train, theta, lam, m)
        flatten_theta = flatten_theta - learning_rate * grad

        theta = [flatten_theta[0:theta[0].size].reshape(theta[0].shape),
                 flatten_theta[theta[0].size:].reshape(theta[1].shape)]

        err_train = (predict(theta, x_train) != y_train).sum()/len(y_train) * 100
        err_test = (predict(theta, x_test) != y_test).sum() / len(y_test) * 100
        error.append((err_train, err_test))

    return theta, error


def learning_curve(epochs, error):
    print(f'Train error: {error[-1][0]}%')
    print(f'Test error: {error[-1][1]}%')

    epoch_c = [x for x in range(epochs)]
    train_acc = [x[0] for x in error]
    test_acc = [x[1] for x in error]

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.plot(epoch_c, train_acc, 'r')
    ax.plot(epoch_c, test_acc, 'g')

    # set chart title and label axes.
    ax.set_title("Learning curve", fontsize=24)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)

    plt.show()
