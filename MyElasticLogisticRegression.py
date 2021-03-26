def generate_batches(X, y, batch_size):
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for i in range(len(X)//batch_size):
        ind = perm[i*batch_size : (i+1)*batch_size]
        yield (X[ind], y[ind])

def logit(x, w):
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

class MyElasticLogisticRegression(object):
    def __init__(self, l1_coef, l2_coef):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def fit(self, X, y, epochs=100, lr=0.5, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        losses = []
        epoch=0
        while epoch < epochs:
            for X_, y_ in generate_batches(X_train, y, batch_size):
                self.w -= lr * self.get_grad(X_, y_, sigmoid(logit(X_,self.w)))
                losses.append(self.__loss(y_, sigmoid(logit(X_,self.w))))
            epoch += 1 
        return losses
    

    def get_grad(self, X_batch, y_batch, predictions):
        wc = np.copy(self.w)
        wc[0] = 0
        grad = X_batch.T @ (predictions - y_batch) / len(X_batch)
        grad += 2 * self.l2_coef * wc
        grad += self.l1_coef * np.sign(wc)

        return grad

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w

    def __loss(self, y, p):  
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
