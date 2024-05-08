import numpy
import numpy as np
from scipy import optimize 


import Util.CG as CG
    
class Executor:
    def __init__(self, xMat, yVec):
        self.s, self.d = xMat.shape
        self.yVec = yVec.reshape(self.s, 1).astype(np.float64)
        self.xMat = xMat * self.yVec
        self.xMat = self.xMat.astype(np.float64)


        # initialize w and p
        self.w = numpy.zeros((self.d, 1)).astype(np.float64)
        self.p = numpy.zeros((self.d, 1)).astype(np.float64)
        

    def setParam(self, gamma, gtol, maxiter, isSearch,  etaList):
        self.gamma = gamma
        self.gtol = gtol
        self.maxiter = maxiter
        self.isSearch = isSearch
        self.etaList = etaList
        self.numEta = len(self.etaList)
        
    def updateP(self, p):
        self.p = p.copy()
        
    def updateW(self):
        self.w -= self.p

    def updateW_(self, w):
        self.w = w.copy()

       
    
    def objFun(self, wVec):
        '''
        f_j (w) = log (1 + exp(-w dot x_j)) + (gamma/2) * ||w||_2^2
        return the mean of f_j for all local data x_j
        '''
        zVec = numpy.dot(self.xMat, wVec.reshape(self.d, 1))
        lVec = numpy.log(1 + numpy.exp(-zVec))
        loss = numpy.mean(lVec)
        reg = self.gamma / 2 * numpy.sum(wVec ** 2)
        return loss + reg
    
    def objFunSearch(self):
        objValVec = numpy.zeros(self.numEta + 1)
        for l in range(self.numEta):
            objValVec[l] = self.objFun( self.w - self.etaList[l] * self.p )
        objValVec[-1] = self.objFun(self.w)
        return objValVec
    
    def computeGrad(self):
        '''
        Compute the gradient of the objective function using local data
        '''
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec)
        vec1 = 1 + expZVec
        vec2 = -1 / vec1
        grad = numpy.mean(self.xMat * vec2, axis=0)
        return grad.reshape(self.d, 1) + self.gamma * self.w

    def computeGrad_(self,w):
        '''
        Compute the gradient of the objective function using local data
        '''
        zVec = numpy.dot(self.xMat, w)
        expZVec = numpy.exp(zVec)
        vec1 = 1 + expZVec
        vec2 = -1 / vec1
        grad = numpy.mean(self.xMat * vec2, axis=0)
        return grad.reshape(self.d, 1) + self.gamma * w    

    def computeNewton_gain(self, gVec, m):

        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)
        

        A = self.xMat * (expZVec / numpy.sqrt(self.s))
        d = A.shape[1]        
        lam = self.gamma
        J = numpy.dot(A.T,A)+lam*numpy.eye(d)
        b = gVec
        print(A.dtype)

        
        b = b.reshape(d, 1)

        p = b   
        basis = []
        for i in range(m):
            p = J @ p
            basis.append( p.copy() )

        basis = numpy.hstack( basis )
        solution  = numpy.linalg.lstsq(basis, b)[0]
        res = numpy.linalg.norm(basis @ solution - b )
        bnorm = numpy.linalg.norm(b)
        cond_number = np.linalg.cond(J)
        print('lam', lam, basis.shape)
        print('cond_number', cond_number)
        s = numpy.sqrt( cond_number ) 
        print('theoretical upper bound',2 * (s - 1)**m / (s + 1)**m)
        print('res:', res)
        print('b:', bnorm)

        sig = numpy.linalg.svd(J, compute_uv=False)
        print('sigular values are ' , sig[0] , sig[-1])

        return res/bnorm, 2 * (s - 1)**m / (s + 1)**m


    def testSY(self, gVec, m, S,Y,eta):

        normft = numpy.linalg.norm(gVec)
        S = S/normft
        Y = Y/normft
        b = gVec/normft

        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)
        

        A = self.xMat * (expZVec / numpy.sqrt(self.s))
        d = A.shape[1]        
        lam = self.gamma
        J = numpy.dot(A.T,A)+lam*numpy.eye(d)
        G = numpy.eye(d) - eta*J

        
        print(A.dtype)

        
        b = b.reshape(d, 1)

        p = b   
        basis = []
        for i in range(m):            
            basis.append( p.copy() )
            p = G @ p

        basis = numpy.hstack( basis )
        Y_limit = J @ basis

        Y_error = numpy.linalg.norm(Y_limit - Y)
        S_error = numpy.linalg.norm(basis - S)
        ASY_error = numpy.linalg.norm(Y - J @ S)

        norms = np.linalg.norm(basis.astype(np.float64), axis=0)
        G_normalized = basis / norms[np.newaxis, :]
        U, Singu, VT = np.linalg.svd(G_normalized.astype(np.float64), full_matrices=False)


        print('Y_error', numpy.linalg.norm(Y_error, axis=0),
                    '\nS_error', numpy.linalg.norm(S_error, axis=0),
                    '\nASY_error', numpy.linalg.norm(ASY_error, axis=0),
                    '\nsingular values', Singu)

        print(np.linalg.norm(G_normalized, axis=0))



        return None

    



        




    
    def computeNewton(self, gVec):
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)

        aMat = self.xMat * (expZVec / numpy.sqrt(self.s))
        
        #pVec = CG.svrgSolver(aMat, gVec, self.gamma, alpha=0.6, Tol=self.gtol, MaxIter=self.maxiter)
        pVec = CG.cgSolver(aMat, gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
        #pVec = CG.cgSolver2(numpy.dot(aMat.T, aMat), gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
        self.gtol *= 0.5 # decrease the convergence error paramter of CG
        
        return pVec

    
    #Here we use AA to replace the original Newton
    def AAP( self,  lr=1, local_epochs = 5 ):

        
        
        x = self.w.copy()
        gVec = self.computeGrad_(x)

        iterations = []
        local_gradients = []        


        for i in range(local_epochs+1):
            gradx = self.computeGrad_(x) 
            print( numpy.linalg.norm(gradx) )

            iterations.append( x.copy() )
            local_gradients.append( gradx.copy()  )

            x = x - lr*( gradx )

        iterations = numpy.hstack( iterations )
        local_gradients = numpy.hstack( local_gradients )


        eta = 1
        S = numpy.diff( iterations )
        Y = numpy.diff( local_gradients ) #Note that Y only contain local epochs
        alpha_ = numpy.linalg.lstsq(Y, gVec)[0]
        pVec =   1 * (eta * gVec + (S - eta * Y) @ alpha_) 

        #print('ls',numpy.linalg.norm(eta * gVec  - (eta * Y) @ alpha_),numpy.linalg.norm(gVec),numpy.linalg.norm(pVec))

        #print( numpy.linalg.norm(gradx),  '-->',  numpy.linalg.norm( self.computeGrad_(self.w - pVec) + grad_gap) )

        print('columnwise_norm: Y',numpy.linalg.norm(Y, axis=0), 'S', numpy.linalg.norm(S, axis=0))
        print('columnwise_norm: iterations', numpy.linalg.norm(iterations, axis=0), 'local_gradients', numpy.linalg.norm(local_gradients, axis=0))

        theta = numpy.linalg.norm(gVec-Y@alpha_) /numpy.linalg.norm(gVec)

        # smallest singular value
        norms = np.linalg.norm(local_gradients.astype(np.float64), axis=0)
        Y_normalized = local_gradients.astype(np.float64) / norms[np.newaxis, :]
        U, Sigu, VT = np.linalg.svd(Y_normalized.astype(np.float64), full_matrices=False)


        newton_gain, theory_gain = self.computeNewton_gain(self.computeGrad(), local_epochs)
        print('newton gain:', newton_gain)
        print('multisecant gain', theta )
        print(' sigmamin(h)',min(Sigu))

        print('pre testing....', np.linalg.norm(gVec-self.computeGrad() ))

        self.testSY(gVec, local_epochs, S,Y, lr) 

       
        
        return pVec,theta,min(Sigu),newton_gain


    #this returns the global updates and local gradient
    def multiGD(self,w, lr=1,local_epochs = 3):
        #print('multiGD LR: ',lr)
        x = w.copy()
        for i in range(local_epochs):
            gradx = self.computeGrad_(x)
            x = x - lr*gradx
        return x, self.computeGrad_(x)







        



        
        