def gradient_descent_update(params, grads, lr):
    for key in params:
        params[key] -= lr * grads[key]