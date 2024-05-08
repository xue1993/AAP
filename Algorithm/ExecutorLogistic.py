import numpy
import numpy as np
from scipy import optimize 

# This code contains many operations to make sure the data type is as required
#Note that numpy.linalg.lstsq and numpy.linalg.svd do not support longdouble, thus the input for them is converted back to np.double

import Util.CG as CG
    
class Executor:
    def __init__(self, xMat, yVec, dtype_=np.double):

        self.dtype_ = dtype_
        self.s, self.d = xMat.shape
        self.yVec = yVec.reshape(self.s, 1).astype(self.dtype_)
        self.xMat = (xMat * self.yVec).astype(self.dtype_) #thus the xMat saves the multiplication between data and label
        self.w = np.zeros((self.d, 1), dtype=self.dtype_)
        self.p = np.zeros((self.d, 1), dtype=self.dtype_)

        

    def setParam(self, gamma, gtol, maxiter, isSearch,  etaList):

        
        if self.dtype_ == np.longdouble:
            self.gamma = np.longdouble(gamma)
            self.gtol = np.longdouble(gtol)
        elif self.dtype_ == np.double: 
            self.gamma = float(gamma)
            self.gtol = float(gtol)
        else:
            self.gamma = gamma
            self.gtol = gtol



        self.maxiter = int(maxiter)
        self.isSearch = isSearch
        self.etaList = np.array(etaList, dtype=self.dtype_)
        self.numEta = len(self.etaList)


    def updateP(self, p):
        self.p = p.copy().astype(self.dtype_)
        
    def updateW(self):
        self.w -= self.p

    def updateW_(self, w):
        self.w = w.copy().astype(self.dtype_)
      
    
    def objFun(self, wVec):
        '''
        f_j (w) = log (1 + exp(-w dot x_j)) + (gamma/2) * ||w||_2^2
        return the mean of f_j for all local data x_j
        '''
        zVec = numpy.dot(self.xMat, wVec.reshape(self.d, 1)).astype(self.dtype_)
        lVec = numpy.log(1 + numpy.exp(-zVec, dtype=self.dtype_))
        loss = numpy.mean(lVec)
        reg = self.gamma / 2 * numpy.sum(wVec ** 2)
        return loss + reg
    
    def objFunSearch(self):
        objValVec = np.zeros(self.numEta + 1, dtype=self.dtype_)
        for l in range(self.numEta):
            objValVec[l] = self.objFun( self.w - self.etaList[l] * self.p )
        objValVec[-1] = self.objFun(self.w)
        return objValVec
    
    def computeGrad(self):
        '''
        Compute the gradient of the objective function using local data
        '''
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        vec1 = 1 + expZVec
        vec2 = -1 / vec1
        grad = numpy.mean(self.xMat * vec2, axis=0)
        return (grad.reshape(self.d, 1) + self.gamma * self.w).astype(self.dtype_)

    def computeGrad_(self,w):
        '''
        Compute the gradient of the objective function using local data
        '''
        w = w.reshape(self.d, 1).astype(self.dtype_)
        zVec = numpy.dot(self.xMat, w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        vec1 = 1 + expZVec
        vec2 = -1 / vec1
        grad = numpy.mean(self.xMat * vec2, axis=0)
        return (grad.reshape(self.d, 1) + self.gamma * w).astype(self.dtype_)    


    #Here we use AA to replace the original Newton
    def AAP( self,  lr=1, local_epochs = 5 ):     


        #intialization
        x = self.w.copy().astype(self.dtype_) 
        gVec = self.computeGrad_(x).astype(self.dtype_) 

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)

        iterations = []
        local_gradients = []        

        #take gd steps
        for i in range(local_epochs+1):
            gradx = self.computeGrad_(x).astype(self.dtype_)  
            print( numpy.linalg.norm(gradx) )

            iterations.append( x.copy() )
            local_gradients.append( gradx.copy()  )

            x = x - lr*( gradx )

        iterations = numpy.hstack( iterations, dtype=self.dtype_ )
        local_gradients = numpy.hstack( local_gradients, dtype=self.dtype_ )


        eta = 1
        S = numpy.diff( iterations ).astype(self.dtype_) 
        Y = numpy.diff( local_gradients ).astype(self.dtype_) 


        #three different ways to solve the LS problem
        #method 1: QR
        Q, R = np.linalg.qr( Y.astype(np.float64) )
        Q_T_b = np.dot( Q.T,  gVec.astype(np.float64) )
        alpha_qr = np.linalg.solve(R, Q_T_b)   

        #method 2: QR with pre_normalization
        b_norm = numpy.linalg.norm(gVec)
        Q, R = np.linalg.qr( (Y/b_norm).astype(np.float64) )
        Q_T_b = np.dot( Q.T,  (gVec/b_norm).astype(np.float64) )
        alpha_normlized = np.linalg.solve(R, Q_T_b)         

        #method 3: np.linalg.lstsq
        alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64))[0].astype(self.dtype_)      
        print( 'norm(solution_lstsq-solutionqr)',  numpy.linalg.norm( alpha_lstsq -  alpha_qr ) , 'qr', numpy.linalg.norm(  alpha_qr ),  'qr_normalization', numpy.linalg.norm( alpha_normlized), 'lstsq', numpy.linalg.norm( alpha_lstsq  ), )
        print( 'AAP LS solution:', alpha_qr.flatten())  

        #Update AAP direction with QR solution
        pVec =   1 * (eta * gVec + (S - eta * Y) @ alpha_qr) 
        print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_qr), 'rtilde', numpy.linalg.norm( Y @ alpha_qr - gVec  )  )

        #LS problem properties
        norms = np.linalg.norm(Y.astype(self.dtype_), axis=0)
        Y_normalized = Y.astype(self.dtype_) / norms[np.newaxis, :]
        U, Sigu, VT = np.linalg.svd(Y_normalized.astype(np.float64), full_matrices=False)
        print('columnwise_norm: Y, b', numpy.linalg.norm(Y, axis=0), b_norm)
        print('sigular values of Y', Sigu)  

        #blow are checks
        #print('ls',numpy.linalg.norm(eta * gVec  - (eta * Y) @ alpha_qr),numpy.linalg.norm(gVec),numpy.linalg.norm(pVec))
        #print( numpy.linalg.norm(gradx),  '-->',  numpy.linalg.norm( self.computeGrad_(self.w - pVec) + grad_gap) )
        print("\n")
        print( 'columnwise_norm: S', numpy.linalg.norm(S, axis=0))
        print('iterations', numpy.linalg.norm(iterations, axis=0), '\nlocal_gradients', numpy.linalg.norm(local_gradients, axis=0))

        # smallest singular value
        norms = np.linalg.norm(local_gradients.astype(self.dtype_), axis=0)
        local_gradients_normalized = local_gradients.astype(self.dtype_) / norms[np.newaxis, :]
        U, Sigu, VT = np.linalg.svd(local_gradients_normalized[:,:-1].astype(np.float64), full_matrices=False)


        #optimization gain
        theta = numpy.linalg.norm(gVec-Y @ alpha_qr) /numpy.linalg.norm(gVec)
        #Newton-GMRES gain
        alpha_Newton, newton_gain, theory_gain = self.computeNewton_gain(self.computeGrad(), local_epochs)
        print( '\nNewton-GMRES solution:', alpha_Newton.flatten())  
        print( '   Length of ptilde:',  numpy.linalg.norm( S @ alpha_Newton), 'rtilde', numpy.linalg.norm( Y @ alpha_Newton - gVec  )  )
        print('\ntheory_Upper_Bound:', theory_gain)
        print('Newton-GMRES gain:', newton_gain)
        print('AAP gain', theta )
        print('sigular values of M_t', Sigu)
        

        #test the covnergence of S Y
        self.testSY(gVec, local_epochs, S,Y, lr)        
        
        return pVec,theta,min(Sigu),newton_gain


    #GD method, aka picard iteration
    def multiGD(self,w, lr=1,local_epochs = 3):
        #print('multiGD LR: ',lr)

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)

        x = w.copy()
        for i in range(local_epochs):
            gradx = self.computeGrad_(x)
            x = x - lr*gradx
        return x, self.computeGrad_(x)



    def computeNewton_gain(self, gVec, m):
        print("\nNewton-GMRES")

        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)     

        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_) )           
        J = numpy.dot(A.T,A)+self.gamma*numpy.eye(self.d).astype(self.dtype_)  #this the hessian matrix      

        
        b = gVec.reshape(self.d, 1).astype(self.dtype_) 
        bnorm = numpy.linalg.norm(b)
        print('residual norm:', bnorm)
        b = b/bnorm #normalize b

        p = b.astype(self.dtype_)   #since the residual gVec is converging to 0, so it's better to normalize it now
        basis = []
        for i in range(m):
            p = J @ p
            basis.append( p.copy() )
        basis = numpy.hstack( basis, dtype=self.dtype_ )
        print('Newton basis: columnwise_norm: ',numpy.linalg.norm(basis, axis=0) )
        print('shape', basis.shape)        
        norms = np.linalg.norm(basis.astype(self.dtype_), axis=0)
        basis_normalized = basis.astype(self.dtype_) / norms[np.newaxis, :]
        U, Sigu, VT = np.linalg.svd(basis_normalized.astype(np.float64), full_matrices=False)
        print('singular values', Sigu)


        #method 1
        solution_lstsq  = numpy.linalg.lstsq(basis.astype(np.float64), b.astype(np.float64))[0]
        
        #method 2
        Q, R = np.linalg.qr(basis.astype(np.float64))
        Q_T_b = np.dot(Q.T, b.astype(np.float64))
        solution_qr = np.linalg.solve(R, Q_T_b)        
        print( 'norm(solution_lstsq-solutionqr)',  numpy.linalg.norm( solution_lstsq -  solution_qr )  )  

        res = numpy.linalg.norm(basis @ solution_qr - b )      
        #print('gain:', res)
        


        cond_number = np.linalg.cond(J.astype(np.float64))       
        sig = numpy.linalg.svd(J.astype(np.float64), compute_uv=False)        
        print('\nHessian info: cond_number :', cond_number)#double compute the condition number to check
        print('sigular values : ' , sig[0],sig[-1])
        print(J.dtype) 
        s = numpy.sqrt( cond_number ) 
        print('theoretical upper bound',2 * (s - 1)**m / (s + 1)**m) #this is the famous bound with condtion number   

        

        return solution_qr, res, 2 * (s - 1)**m / (s + 1)**m


    def testSY(self, gVec, m, S,Y,lr):
        print("\nSY Convergence")
        print('lr', lr)
        print( 'S[0]-lr*f_t', numpy.linalg.norm(S[:,0].reshape(self.d, 1) + lr* gVec.reshape(self.d, 1) ) )

  
        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)

        #normalize S Y b, otherwise all--> 0
        normft = numpy.linalg.norm(gVec.astype(self.dtype_))         
        b = (gVec.astype(self.dtype_)/normft).reshape(self.d, 1) 
        S = S.astype(self.dtype_)/normft
        Y = Y.astype(self.dtype_)/normft
        print('(S[0]-lr*f_t)/|f_t|(floating error occur):',numpy.linalg.norm(S[:,0].reshape(self.d, 1) + lr* b )  )


        #below we construct the limit matrix of S and Y, both normalize
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)    
        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_))       
        J = numpy.dot(A.T,A)+ self.gamma *numpy.eye(self.d).astype(self.dtype_) 
        G = numpy.eye(self.d) - lr*J #g'(x_t)

        p = b.copy()   
        basis = []
        for i in range(m):            
            basis.append( p.copy() )
            p = G @ p #note the difference between newton method

        basis = -lr*numpy.hstack( basis, dtype=self.dtype_ ) #this is the theoretical limit of S
        Y_limit = J @ basis

        Y_error =Y_limit - Y
        S_error = basis - S
        ASY_error = Y - J @ S

        norms = np.linalg.norm(basis.astype(self.dtype_), axis=0)
        G_normalized = basis / norms[np.newaxis, :]
        U, Singu, VT = np.linalg.svd(G_normalized.astype(np.float64), full_matrices=False)


        print('Y_error', numpy.linalg.norm(Y_error, axis=0),
                    '\nS_error', numpy.linalg.norm(S_error, axis=0),
                    '\nASY_error', numpy.linalg.norm(ASY_error, axis=0),
                    '\nsingular values of G_t', Singu)

        return None


    
    def computeNewton(self, gVec):
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)

        aMat = self.xMat * (expZVec / numpy.sqrt(self.s))
        
        #pVec = CG.svrgSolver(aMat, gVec, self.gamma, alpha=0.6, Tol=self.gtol, MaxIter=self.maxiter)
        pVec = CG.cgSolver(aMat, gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
        #pVec = CG.cgSolver2(numpy.dot(aMat.T, aMat), gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
        self.gtol *= 0.5 # decrease the convergence error paramter of CG
        
        return pVec

    
    







        



        
        