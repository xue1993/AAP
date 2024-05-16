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
        self.stop = False
        self.iterations_history = []
        self.local_gradients_history = []

        

    def setParam(self, gamma,  isSearch,  etaList):

        
        if self.dtype_ == np.longdouble:
            self.gamma = np.longdouble(gamma)
        elif self.dtype_ == np.double: 
            self.gamma = float(gamma)
        else:
            self.gamma = gamma


        self.damping = 1 #damping ratio
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

    #Picard, multiple GD steps
    def Picard( self,  lr=1, local_epochs = 5 ):     


        #intialization
        x = self.w.copy().astype(self.dtype_) 

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)
       

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            gradx = self.computeGrad_(x).astype(self.dtype_)  
            print( numpy.linalg.norm(gradx) )
            x = x - lr*( gradx )
        
        return self.w-x


    #AAP method
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

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            gradx = self.computeGrad_(x).astype(self.dtype_)  
            print( numpy.linalg.norm(gradx) )

            iterations.append( x.copy() )
            local_gradients.append( gradx.copy()  )

            x = x - lr*( gradx )

        iterations = numpy.hstack( iterations, dtype=self.dtype_ )
        local_gradients = numpy.hstack( local_gradients, dtype=self.dtype_ )


        S = numpy.diff( iterations ).astype(self.dtype_) 
        Y = numpy.diff( local_gradients ).astype(self.dtype_) 
        print('S.shape:  ', S.shape)
 
        #solve the unconstrained LS problem
        alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
      
        #Update AAP direction with alpha_lstsq solutin
        pVec =   self.damping *lr* gVec + (S - self.damping *lr* Y) @ alpha_lstsq 
        print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )
        
        return pVec

    
    #Classical AA(m) method, the different is in each global iteration, we take m+1 AA step
    def AA( self,  lr=1, m = 5 ):     


        #intialization
        x = self.w.copy().astype(self.dtype_)         

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)

        for i in range(m+1):
            gVec = self.computeGrad_(x).astype(self.dtype_)
            print( numpy.linalg.norm(gVec) )

            #Update the history points
            self.iterations_history.append( x.copy() )
            self.local_gradients_history.append( gVec.copy()  )

            if len( self.iterations_history )> (m+1):
                self.iterations_history.pop(0)
                self.local_gradients_history.pop(0)               

                
            #Update S Y matrix
            iterations = numpy.hstack( self.iterations_history, dtype=self.dtype_ )
            local_gradients = numpy.hstack( self.local_gradients_history, dtype=self.dtype_ )

            S = numpy.diff( iterations ).astype(self.dtype_) 
            Y = numpy.diff( local_gradients ).astype(self.dtype_) 
            print('S.shape:  ', S.shape)
    
            #solve the unconstrained LS problem
            alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   self.damping * lr *gVec + (S - self.damping *lr *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            x -= pVec
        
        return self.w-x


    #resAA method
    def resAA( self,  lr=1, m = 5 ):     


        #intialization
        x = self.w.copy().astype(self.dtype_) 

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)

        iterations_list = []
        local_gradients_list = []        

        #take gd steps
        for i in range(m+1):
            gVec = self.computeGrad_(x).astype(self.dtype_)  
            print( numpy.linalg.norm(gVec) )

            iterations_list.append( x.copy() )
            local_gradients_list.append( gVec.copy()  )
            
            iterations = numpy.hstack( iterations_list, dtype=self.dtype_ )
            local_gradients = numpy.hstack( local_gradients_list, dtype=self.dtype_ )


            S = numpy.diff( iterations ).astype(self.dtype_) 
            Y = numpy.diff( local_gradients ).astype(self.dtype_) 
            print('S.shape:  ', S.shape)
 
            #solve the unconstrained LS problem
            alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   self.damping *lr* gVec + (S - self.damping *lr *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            x -= pVec
        
        return self.w-x




    def Newton_GMRES(self, m):
    #This is a fake Newton-GMRES solver,

        gVec = self.computeGrad().astype(self.dtype_)

        #step 1, generate the Jacobian J
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec) 
        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_) )           
        J = numpy.dot(A.T,A)+self.gamma*numpy.eye(self.d).astype(self.dtype_)  #this the hessian matrix      

        #step 2, normalize the b matrix to to aviod too small values of b
        b = gVec.reshape(self.d, 1).astype(self.dtype_) 
        bnorm = numpy.linalg.norm(b)
        b = b/bnorm #normalize b

        #step 3, generate the Krylov sequence Af_t, A^2f_t,... Note that it is different with the true krylov subspace f_t, Af_t,A^2f_t,...
        p = b.astype(self.dtype_)   #since the residual gVec is converging to 0, so it's better to normalize it now
        basis = []  #this basis is with A_t
        for i in range(m):
            p = J @ p
            basis.append( p.copy() )
        basis = numpy.hstack( basis, dtype=self.dtype_ )
        print(basis.shape)


        #step 4, solve the projection Ap. the solution stability of influenced by the conditon number, but the condition number could be very small with large m.
        solution_lstsq  = numpy.linalg.lstsq(basis.astype(np.float64), b.astype(np.float64),rcond=None)[0]
        


        #step 5,  we know the projection Ap as basis @ solution_lstsq, we need to solve the real p
        r_ = (bnorm*basis @ solution_lstsq).astype(np.float64)
        # solve Ap = (Ap)
        pVec = np.linalg.solve( J.astype(np.float64), r_ )       
        print( 'Length of p:',  numpy.linalg.norm( pVec ) ) 
      

        return  pVec

    def Newton_CG(self, m):

        gVec = self.computeGrad().astype(self.dtype_)

        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec) 
        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_) )           
        J = numpy.dot(A.T,A)+self.gamma*numpy.eye(self.d).astype(self.dtype_)

        
        pVec = CG.cgSolver_J(J, gVec,  MaxIter=m)   
        print( 'Length of p:',  numpy.linalg.norm( pVec ) )   
        
        return pVec



    

    
    







        



        
        