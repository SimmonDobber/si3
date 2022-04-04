import numpy as np

class HMM:
    def __init__(self, observed, transition_matrix, emission_matrix, initial_distribution):
        self.I = initial_distribution
        self.V = np.array(observed)
        self.A = np.array(transition_matrix)
        self.B = np.array(emission_matrix)

        self.K = self.A.shape[0]
        self.N = self.V.shape[0]

    def forward(self):
        alpha = np.zeros((self.N, self.K))
        alpha[0, :] = self.I * self.B[:, self.V[0]]
        for t in range(1, self.V.shape[0]):
            for j in range(self.A.shape[0]):
                alpha[t, j] = alpha[t - 1].dot(self.A[:, j]) * self.B[j, self.V[t]]
        return np.argmax(alpha, axis=1), alpha

    def backward(self):
        beta = np.zeros((self.N, self.K))
        beta[self.V.shape[0] - 1] = np.ones((self.A.shape[0]))
        for t in range(self.V.shape[0] - 2, -1, -1):
            for j in range(self.A.shape[0]):
                beta[t, j] = (beta[t + 1] * self.B[:, self.V[t + 1]]).dot(self.A[j, :])
        return np.argmax(beta, axis=1), beta

    def forward_backward(self):
        fbv = self.forward()[1] * self.backward()[1]
        return np.argmax(fbv, axis=1)

    def viterbi(self):
        K = self.A.shape[0]
        T = len(self.V)
        T1 = np.empty((K, T), 'd')
        T2 = np.empty((K, T), 'B')

        T1[:, 0] = self.I * self.B[:, self.V[0]]
        T2[:, 0] = 0

        for i in range(1, T):
            T1[:, i] = np.max(T1[:, i - 1] * self.A.T * self.B[np.newaxis, :, self.V[i]].T, 1)
            T2[:, i] = np.argmax(T1[:, i - 1] * self.A.T, 1)

        x = np.empty(T, 'B')
        x[-1] = np.argmax(T1[:, T - 1])
        for i in reversed(range(1, T)):
            x[i - 1] = T2[x[i], i]
        return x





