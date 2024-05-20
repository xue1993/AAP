import numpy as np
import sys

# Append the home directory to system path for importing custom modules
home_dir = '../'
sys.path.append(home_dir)
from Algorithm.ExecutorNMF_WN import Executor as NMF
EtaList = 1 / (4 ** np.arange(0, 10))

class Solver:
    
    def __init__(self, local_epochs=5, eta0=1.0, C=1.0, dtype_ = np.double, algo_='AAP'):

        self.tolConverge = 1e-13
        self.eta0 = float(eta0)
        self.decay_rate = float(C)
        self.local_epochs = int(local_epochs)
        self.executor = None
        self.dtype_ = dtype_
        self.algo = algo_ #choices: AAP, Newton ,Newton-CG,Newton-GMRES, AA, resAA
        #self.problem = problem_
        

    def fit(self, A, k, W0=None, H0=None):
        """
        pass the data
        """
        #below is pass the logistic regression data with optional 
        self.m,self.n = A.shape
        self.k = k
        self.W = W0.copy() if W0 is not None else np.random.rand(self.m, self.k).astype(self.dtype_)        
        self.H = H0.copy() if H0 is not None else np.zeros((self.k, self.n)).astype(self.dtype_)

        self.executor = NMF(A, k, self.W, self.H, dtype_= self.dtype_)
        

        

    def train(self, maxIter=20, isSearch=False, warmup=True, damping=.1):   
        

        self.etaList = EtaList
        self.numEta = len(self.etaList)        
        self.executor.setParam(None, isSearch, self.etaList, warmup_=warmup, damping_=damping)   #warm up included

        W = self.executor.W.copy()    
        H = self.executor.H.copy()  

        self.errorList = []
        err = self.executor.objFun(W,H)
        self.errorList.append(  err  )

        self.thetaList = []
        self.newtongainList = []
        self.sigmaList = []

        for t in range(maxIter):
            print(f"\n============== Iteration {t+1}: ====error={err}=========")
            

            if  self.algo == 'AAP':
                pW,pH =  self.executor.AAP(lr=self.eta0, local_epochs=self.local_epochs)
            elif self.algo == 'AA':
                pW,pH =  self.executor.AA(lr=self.eta0, m=self.local_epochs)
            elif self.algo == 'resAA':
                pW,pH =  self.executor.resAA(lr=self.eta0, m=self.local_epochs)
            else: 
                pW,pH =  self.executor.Picard(lr=self.eta0, local_epochs=self.local_epochs)
               
            
            self.executor.updateP(pW,pH)            
            self.executor.updateWH()
            W -= pW
            H -= pH
            
            
            err = self.executor.objFun(W, H)
            self.errorList.append(err)
            if err < self.tolConverge or np.isnan(W).any() or np.isnan(H).any() or self.executor.stop:
                print(f"Iteration {t}: Convergence achieved, or stop={ self.executor.stop}.")
                break

        self.funcValsPerPicard = self.executor.funcVals

        return self.errorList