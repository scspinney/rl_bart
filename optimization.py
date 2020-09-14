import numpy as np


class Adam():

    def __init__(self,lr=0.0001,n_weights=11, n_iter=10, beta1=0.9, beta2=0.999, epsilon=1e-8, delta=1e-3):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(n_weights)
        self.v = np.zeros(n_weights)
        self.delta = delta
        self.t = 0

    def backwards(self,weights,gradients):

        self.t+=1
        for k in range(len(gradients)):

            self.m[k] = self.beta1*self.m[k] + (1.-self.beta1)*gradients[k]
            self.v[k] = self.beta2*self.v[k] + (1.-self.beta2)*(gradients[k]**2)
            m_hat = self.m[k]/(1.-self.beta1**self.t)
            v_hat = self.v[k]/(1.-self.beta2**self.t)
            weights[k] += self.lr*m_hat/(np.sqrt(v_hat) + self.epsilon)

        return weights


