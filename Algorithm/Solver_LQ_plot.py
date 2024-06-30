import numpy as np
import sys

# Append the home directory to system path for importing custom modules
home_dir = '../'
sys.path.append(home_dir)
from Algorithm.ExecutorLogistic_plot import Executor as  Logistic
EtaList = 1 / (4 ** np.arange(0, 10))

class Solver:
    
    def __init__(self, local_epochs=5, eta0=1.0, C=1.0, dtype_ = np.double, algo_='AAP', problem_='Quadratic'):

        self.tolConverge = 1e-13
        self.eta0 = float(eta0)
        self.decay_rate = float(C)
        self.local_epochs = int(local_epochs)
        self.executor = None
        self.dtype_ = dtype_
        self.algo = algo_ #choices: AAP, Newton ,Newton-CG,Newton-GMRES, AA, resAA
        self.problem = problem_
        

    def fit(self, xMat, yVec):
        """
        pass the data
        """
        #below is pass the logistic regression data with optional 
        n, d = xMat.shape
        if self.problem == 'LogisticTest':
            print('LogisticTest activated')
            self.executor = Logistic_test(xMat, yVec, dtype_= self.dtype_)
        elif self.problem == 'QuadraticTest':
            print('QuadraticTest activated')
            self.executor = Logistic_test(xMat, yVec, dtype_= self.dtype_)
        elif self.problem == 'Q':
            print('Quadratic activated')
            self.executor = Quadratic(xMat, yVec, dtype_= self.dtype_)
        else:
            print('Logistic activated')
            self.executor = Logistic(xMat, yVec, dtype_= self.dtype_)
        self.n, self.d = n, d

    def train(self, gamma, wopt, maxIter=20, isSearch=False,warmUp=False):
        

        wnorm = np.linalg.norm(wopt.astype(np.float64))
        w = np.zeros((self.d, 1), dtype=self.dtype_)
        self.errorList = []
        err = 1
        self.errorList.append(err)

        self.thetaList = []
        self.newtongainList = []
        self.sigmaList = []

        self.etaList = EtaList
        self.numEta = len(self.etaList)
        
        self.executor.setParam(gamma, wopt, isSearch, self.etaList)
        if warmUp:
            self.executor.warmUp(lr = self.eta0 )
            w = self.executor.w.copy()

        for t in range(maxIter):
            print(f"\n============== Iteration {t+1}: ====error={np.linalg.norm(w - wopt)}=========")
            w_old = w.copy()

            if  self.problem == 'Logistic_test':
                p, theta, sigma, newton_gain = self.executor.AAP(lr=self.eta0, local_epochs=self.local_epochs)   
                self.thetaList.append(theta)
                self.newtongainList.append(newton_gain)
                self.sigmaList.append( sigma )
            elif self.algo == 'AAP':
                p =  self.executor.AAP(lr=self.eta0, local_epochs=self.local_epochs)
            elif self.algo == 'AA':
                p =  self.executor.AA(lr=self.eta0, m=self.local_epochs)
            elif self.algo == 'resAA':
                p =  self.executor.resAA(lr=self.eta0, m=self.local_epochs)
            elif self.algo == 'Newton_CG':
                p =  self.executor.Newton_CG(m=self.local_epochs)
            elif self.algo == 'Newton_GMRES':
                p =  self.executor.Newton_GMRES( m=self.local_epochs)
            else: 
                p =  self.executor.Picard(lr=self.eta0, local_epochs=self.local_epochs)
               
            
            self.executor.updateP(p)            
            self.executor.updateW()
            w -= p
            err = np.linalg.norm(w - wopt) / wnorm
            self.errorList.append(err)
            if err < self.tolConverge or np.isnan(w).any() or self.executor.stop:
                print(f"Iteration {t}: Convergence achieved, or stop={ self.executor.stop}.")
                break

        self.errorPerPicard = self.executor.errorPerPicard

        return self.errorList