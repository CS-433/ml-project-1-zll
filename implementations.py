import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    return np.mean(e**2) / 2


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - np.dot(tx, w)
    grad = -np.dot(tx.T, e) / len(e)
    return grad


def mean_squared_error_gd(
    y, tx, initial_w, max_iters, gamma
):  # first function required
    """The Gradient Descent (GD) algorithm using mean squared error.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) of the last iteration of GD
        w: the model parameters of last iteration of GD
    """
    w = initial_w
    losses = []
    loss = compute_loss_mse(y, tx, w)
    losses.append(loss)
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss_mse(y, tx, w)
        losses.append(loss)
    print("Average loss with Gradient Descent(GD): ", np.mean(losses))
    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.
    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - np.dot(tx, w)
    grad = -np.dot(tx.T, e) / len(e)
    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(
    y, tx, initial_w, max_iters, gamma
):  # second function required
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) of the last iteration of SGD
        w: the model parameters of last iteration of SGD
    """
    w = initial_w
    losses = []
    loss = compute_loss_mse(y, tx, w)
    losses.append(loss)
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = compute_loss_mse(minibatch_y, minibatch_tx, w)
            losses.append(loss)
    print("Average loss with Stochastic Gradient Descent(SGD): ", np.mean(losses))
    return w, loss


def least_squares(y, tx):  # third function required
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss value (scalar) of the least squares solution
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y - np.dot(tx, w)
    loss = np.mean(e**2) / 2
    print("Loss with Least Square: ", loss)
    return w, loss


def ridge_regression(y, tx, lambda_):  # fourth function required
    """implement ridge regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss value (scalar) of the ridge regression
    """
    lam = 2 * np.shape(tx)[0] * lambda_
    w = np.linalg.solve(tx.T.dot(tx) + lam * np.eye(np.shape(tx)[1]), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    print("Loss with Ridge Regression: ", loss)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))


def calculate_loss_log(y, tx, w):
    """compute the cost by negative log likelihood.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1). The vector of model parameters.
    Returns:
        a non-negative loss, corresponding to the input parameters w.
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    sum = 0
    N = np.shape(tx)[0]
    for i in range(N):
        sum = (
            sum
            + y[i] * np.log(sigmoid(tx[i].dot(w)))
            + (1 - y[i]) * np.log(1 - sigmoid(tx[i].dot(w)))
        )
    # return float(- sum / N)
    return np.asanyarray(-sum / N)


def calculate_gradient_log(y, tx, w):
    """compute the gradient of loss.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1).  The vector of model parameters.
    Returns:
        a vector of shape (D, 1), containing the gradient of the loss at w.
    """
    n = np.shape(tx)[0]
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / n


def logistic_regression(y, tx, initial_w, max_iters, gamma):  # fifth function required
    """
    Do gradient descent using logistic regression. Return the loss and the optimal weights w.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters: a scalar denoting the total number of iterations of logistic_regression
        gamma: float
    Returns:
        loss: scalar number, the loss value (scalar) of the last iteration of logistic regression
        w: shape=(D, 1), the model parameters of last iteration of logistic regression
    """
    threshold = 1e-8
    w = initial_w
    losses = []
    loss = calculate_loss_log(y, tx, w)
    losses.append(float(loss))
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w = w - gamma * calculate_gradient_log(y, tx, w)
        loss = calculate_loss_log(y, tx, w)
        # converge criterion
        losses.append(float(loss))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print(
        "Average loss with Gradient Descent(GD) using Logistic Regression: ",
        np.mean(losses),
    )
    # visualization    TBD

    return w, loss


def calculate_hessian_log(y, tx, w):
    """return the Hessian of the loss function.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a hessian matrix of shape=(D, D)
    """
    n = np.shape(tx)[0]
    s = np.zeros((n, n))
    for i in range(n):
        s[i, i] = sigmoid(tx[i].dot(w)) * (1 - sigmoid(tx[i].dot(w)))
    return (tx.T.dot(s)).dot(tx) / n


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma
):  # sixth function required
    """
    Do gradient descent, using the penalized logistic regression. Return the loss and the optimal weights w.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of logistic_regression
        gamma: scalar
        lambda_: scalar
    Returns:
        loss: scalar number, the loss value (scalar) of the last iteration of penalized logistic regression
        w: shape=(D, 1), the model parameters of last iteration of penalized logistic regression
    """
    w = initial_w
    threshold = 1e-8
    losses = []
    loss = calculate_loss_log(y, tx, w)
    losses.append(float(loss))
    for iter in range(max_iters):
        # get loss and update w.
        gradient = calculate_gradient_log(y, tx, w) + lambda_ * 2 * w
        w = w - gamma * gradient
        loss = calculate_loss_log(y, tx, w)
        # converge criterion
        losses.append(float(loss))
    # if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
    #    break
    print(
        "Average loss with Gradient Descent(GD) using Regularized Logistic Regression: ",
        np.mean(losses),
    )
    # visualization   TBD

    return w, loss


def predict_labels(w, tx):
    """Generates class predictions given optimal weights and a test feature matrix"""
    y_label = tx.dot(w)
    y_label[np.where(y_label <= 0)] = -1
    y_label[np.where(y_label > 0)] = 1
    return y_label
