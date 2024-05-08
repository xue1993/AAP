import numpy as np
import sys

# Append the home directory to system path for importing custom modules
home_dir = '../'
sys.path.append(home_dir)
from Algorithm.ExecutorLogistic import Executor

EtaList = 1 / (4 ** np.arange(0, 10))

class Solver:
    
    def __init__(self, local_epochs=5, eta0=1.0, C=1.0, dtype_ = np.double):

        self.tolConverge = 1e-13
        self.eta0 = float(eta0)
        self.decay_rate = float(C)
        self.local_epochs = int(local_epochs)
        self.executorList = []
        self.dtype_ = dtype_

    def fit(self, xMat, yVec):
        """
        Fit model by partitioning data and initializing executors.
        
        Parameters:
        xMat (numpy.ndarray): Matrix of input features.
        yVec (numpy.ndarray): Vector of target labels.
        """
        n, d = xMat.shape
        #perm = np.random.permutation(n)
        #xMat, yVec = xMat[perm, :], yVec[perm, :]
        self.executorList.append(Executor(xMat, yVec, dtype_= self.dtype_))
        self.n, self.d = n, d

    def train(self, gamma, wopt, maxIter=20, isSearch=False, newtonTol=1e-100, newtonMaxIter=20):

        wnorm = np.linalg.norm(wopt)
        w = np.zeros((self.d, 1), dtype=np.float64)
        self.errorList = []
        self.thetaList = []
        self.newtongainList = []
        self.sigmaList = []

        self.etaList = EtaList
        self.numEta = len(self.etaList)
        
        for executor in self.executorList:
            executor.setParam(gamma, newtonTol, newtonMaxIter, isSearch, self.etaList)


        err = 1
        self.errorList.append(err)

        for t in range(maxIter):
            print(f"\n============== Iteration {t+1}: ====err={err}=========")
            w_old = w.copy()

            executor = self.executorList[0]
            p, theta, sigma, newton_gain = executor.AAP(lr=self.eta0, local_epochs=self.local_epochs)
            executor.updateP(p)            
            executor.updateW()
            w -= p
            
            err = np.linalg.norm(w - wopt) / wnorm
            self.errorList.append(err)
            self.thetaList.append(theta)
            self.newtongainList.append(newton_gain)
            self.sigmaList.append( sigma )
            
            if err < self.tolConverge or np.isnan(w).any():
                print(f"Iteration {t}: Convergence achieved or numerical instability detected.")
                break

        return self.errorList