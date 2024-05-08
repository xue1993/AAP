import numpy
import numpy as np
import sys
home_dir = '../'
sys.path.append(home_dir)
from Algorithm.ExecutorLogistic import Executor

EtaList = 1 / (4 ** numpy.arange(0, 10))

class Solver:
    def __init__(self, local_epochs = 5, eta0=1, C=1):

        self.tolConverge = 1e-13
        self.eta0 = eta0
        self.decayrate  = C
        self.local_epochs = local_epochs
        self.executorList = list()
    
    
    def fit(self, xMat, yVec):
        '''
        Partition X and y to self.m blocks.
        If s is not given, then we set s=n/m and the partition has no overlap.
        '''
        n, d = xMat.shape
        perm = numpy.random.permutation(n)
        xMat = xMat[perm, :]
        yVec = yVec[perm, :]


        executor = Executor(xMat, yVec)
        self.executorList.append(executor)           
        
        self.n = n
        self.d = d
        
    
    def train(self, gamma, wopt, maxIter=20, isSearch=False, newtonTol=1e-100, newtonMaxIter=20):
        print( 'the norm of global minimizer is ', numpy.linalg.norm(wopt) )
        cosList = list()
        errorList = list()
        self.sigmaList = list() 
        self.thetaList = list()          
        self.newtongainList = list()
        wnorm = numpy.linalg.norm(wopt)
        w = numpy.zeros((self.d, 1)).astype(np.float64)
            
        err = numpy.linalg.norm(w - wopt) / wnorm
        errorList.append(err)
        
        self.etaList = EtaList
        self.numEta = len(self.etaList)
        
        for executor in self.executorList:
            executor.setParam(gamma, newtonTol, newtonMaxIter, isSearch, self.etaList)
        
        # iteratively update w
        for t in range(maxIter):
            wold = w.copy()


            #compute Newton direction
            executor = self.executorList[0]

            p, theta,sigma ,newton_gain= executor.AAP(lr=self.eta0/(1+self.decayrate*t), local_epochs=self.local_epochs )
            self.sigmaList.append( sigma  )
            self.thetaList.append( theta)
            self.newtongainList.append( newton_gain)


            print('the  gradient norm is', numpy.linalg.norm(executor.computeGrad()), ' the p norm is ', numpy.linalg.norm(p))


            executor.updateP(p)            
            executor.updateW()

            # driver takes a Newton step
            w -= p

            
            err = numpy.linalg.norm(w - wopt) / wnorm
            errorList.append(err)
            print('Iter ' + str(t) + ': error is ' + str(err))

            
            cos = numpy.dot((wold - wopt).T, wold - w) / (numpy.linalg.norm(wold - wopt) * numpy.linalg.norm(wold - w))
            cosList.append(cos[0][0])
            
            diff = numpy.linalg.norm(w - wold)
            #print('The change between two iterations is ' + str(diff))
            if diff < self.tolConverge or numpy.isnan(w).any():
               break
        
        self.w = w
        print('cos: ',cosList)
        return errorList


 