import numpy
import numpy as np
from scipy import optimize 

from scipy.linalg import toeplitz

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
        self.Tol = 1e-13

        self.gradnorm_t = []
        print(' self.gradnorm_t intilized ')       

        self.errorPerPicard = []
        self.wopt = np.zeros((self.d, 1), dtype=self.dtype_)
        self.woptnorm =1  

        self.Newton_gains = []
        self.Theory_gains = []
        self.AAP_gains = []          
        

    def setParam(self, gamma, wopt, isSearch,  etaList):

        
        if self.dtype_ == np.longdouble:
            self.gamma = np.longdouble(gamma)
        elif self.dtype_ == np.double: 
            self.gamma = float(gamma)
        else:
            self.gamma = gamma

        self.wopt = wopt.astype(self.dtype_)
        self.woptnorm = np.linalg.norm(self.wopt.astype(np.float64))
        self.errorPerPicard.append(  np.linalg.norm(self.w - self.wopt) / self.woptnorm )

        self.path = []
        self.path.append( self.w.copy() )

        #self define two directions and normalize
        self.d1 = self.w - self.wopt
        self.d2 = self.computeGrad()
        self.d1 = self.d1 / np.linalg.norm(self.d1)
        self.d2 = self.d2 / np.linalg.norm(self.d2)

        #contour
        # Create a grid for visualization
        grid_size = 50
        u = np.linspace(-2, 2, grid_size)
        v = np.linspace(-2, 2, grid_size)
        u, v = np.meshgrid(u, v)
        self.loss_values = np.zeros_like(u)

        # Calculate the loss values over the grid
        '''
        for i in range(grid_size):
            for j in range(grid_size):
                params = self.w + u[i, j] * self.d1 + v[i, j] * self.d2
                self.loss_values[i, j] = self.objFun(params)
        '''


        self.damping = 1 #damping ratio
        self.isSearch = isSearch
        self.etaList = np.array(etaList, dtype=self.dtype_)
        self.numEta = len(self.etaList)

        

    def warmUp(self,  lr=1 ):     


        #intialization
        print('WarmUp activated')
        x = self.w.copy().astype(self.dtype_) 

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)
       

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(5):
            gradx = self.computeGrad_(x).astype(self.dtype_)  

            #Update the history points
            self.iterations_history.append( x.copy() )
            self.local_gradients_history.append( gradx.copy()  )

            print( numpy.linalg.norm(gradx) )
            x = x - lr*( gradx )
            self.errorPerPicard.append(  np.linalg.norm(x- self.wopt) / self.woptnorm )

        self.w = x


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

    def computeNewton_gain(self,  m):
        print("\nNewton-GMRES")

        #step 1, generate the Jacobian J
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec) 
        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_) )           
        J = numpy.dot(A.T,A)+self.gamma*numpy.eye(self.d).astype(self.dtype_)    

        
        b = self.computeGrad().reshape(self.d, 1).astype(self.dtype_) 
        bnorm = numpy.linalg.norm(b)
        print('residual norm:', bnorm)
        b = b/bnorm #normalize b

        #generate the Krylov sequence Af_t, A^2f_t,... Note that it is different with the true krylov subspace f_t, Af_t,A^2f_t,...
        p = b.astype(self.dtype_)   #since the residual gVec is converging to 0, so it's better to normalize it now
        basis = []  #this basis is with A_t
        for i in range(m):
            p = J @ p
            basis.append( p.copy() )
        basis = numpy.hstack( basis, dtype=self.dtype_ )
        print('Newton base shape', basis.shape)        


        #get the projection of AK_m(A, f_t)
        solution_lstsq  = numpy.linalg.lstsq(basis.astype(np.float64), b.astype(np.float64))[0] 

        res = numpy.linalg.norm(basis @ solution_lstsq - b )    
        print( 'GMRES gain:', res)        


        cond_number = np.linalg.cond(J.astype(np.float64))              
        print('\nHessian info: cond_number :', cond_number)
        print(J.dtype) 
        s = numpy.sqrt( cond_number ) 
        print('GMRES gain theoretical upper bound',2 * (s - 1)**m / (s + 1)**m) #this is the famous bound with condtion number   


        return res, 2 * (s - 1)**m / (s + 1)**m 


    def Newton_info(self, w, w_old):
    #This is a fake Newton-GMRES solver

        #step 1, generate the Jacobian J
        zVec = numpy.dot(self.xMat, w)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec) 
        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_) )           
        J = numpy.dot(A.T,A)+self.gamma*numpy.eye(self.d).astype(self.dtype_)  

        cond_number = np.linalg.cond(J.astype(np.float64))
        print('cond number', cond_number)

        sig = numpy.linalg.svd(J.astype(np.float64), compute_uv=False)
        print( sig )
        print('sig,', sig[0], sig[-1])

        #step 2, generate the Jacobian J for w_old
        zVec = numpy.dot(self.xMat, w_old)
        expZVec = numpy.exp(zVec, dtype=self.dtype_)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec) 
        A = self.xMat * (expZVec / numpy.sqrt(self.s, dtype=self.dtype_) )           
        J_old = numpy.dot(A.T,A)+self.gamma*numpy.eye(self.d).astype(self.dtype_)

        print('Jacobian Lipschiz', np.linalg.norm(J-J_old)/np.linalg.norm(w-w_old))



    def print_alpha(self, x):
        
        m = x.size
        y = np.zeros(m+1)

        y[0] = -x[0]
        for i in range(m-1):
            y[i+1] =  x[i] - x[i+1]
        y[-1] = x[-1] -1


        print('alpha values:',  y)



    #Picard, multiple GD steps
    def Picard( self,  lr=1, local_epochs = 5 ):    
        
        self.gradnorm_t.append(   numpy.linalg.norm( self.computeGrad().astype(self.dtype_)  )  ) 


        #intialization
        x = self.w.copy().astype(self.dtype_) 

        if self.dtype_ == np.longdouble:
            lr = np.longdouble(lr)
        if self.dtype_ == np.double:  
            lr = float(lr)
       

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            gradx = self.computeGrad_(x).astype(self.dtype_)  
            gradxnorm =  numpy.linalg.norm(gradx) 
            print( gradxnorm )

            if gradxnorm  < self.Tol:
                self.stop = True
            x = x - lr*( gradx )
            self.errorPerPicard.append(  np.linalg.norm(x- self.wopt) / self.woptnorm )
        
        return self.w-x


    #AAP method
    def AAP( self,  lr=1, local_epochs = 5 ):    
        
        self.gradnorm_t.append(   numpy.linalg.norm( self.computeGrad().astype(self.dtype_)  )  ) 


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
            gradxnorm =  numpy.linalg.norm(gradx) 
            print( gradxnorm )

            if gradxnorm  < self.Tol:
                self.stop = True

            iterations.append( x.copy() )
            local_gradients.append( gradx.copy()  )

            x = x - lr*( gradx )
            self.path.append( x.copy() )
            self.errorPerPicard.append(  np.linalg.norm(x- self.wopt) / self.woptnorm )

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
        self.errorPerPicard[-1] = np.linalg.norm(self.w - pVec- self.wopt) / self.woptnorm
        self.path[-1] = self.w - pVec

        #optimization gain        
        newton_gain, theory_gain = self.computeNewton_gain(local_epochs)
        theta = numpy.linalg.norm(gVec-Y @ alpha_lstsq) /numpy.linalg.norm(gVec)
        print('AAP gain', theta )
        self.Newton_gains.append( newton_gain )
        self.Theory_gains.append( theory_gain )
        self.AAP_gains.append( theta )
        
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
            gradxnorm =  numpy.linalg.norm(gVec) 
            print( gradxnorm )

            if gradxnorm  < self.Tol:
                self.stop = True

            #Update the history points
            self.iterations_history.append( x.copy() )
            self.local_gradients_history.append( gVec.copy()  )

            while len( self.iterations_history )> (m+1):
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
            self.path.append( x.copy() )
            self.errorPerPicard.append(  np.linalg.norm(x- self.wopt) / self.woptnorm )
        
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

            x_old = x.copy()

            gVec = self.computeGrad_(x).astype(self.dtype_)  
            gradxnorm =  numpy.linalg.norm(gVec) 
            print( gradxnorm )

            if gradxnorm  < 1e-8:
                self.stop = True

            #print the angle between gradient and the true direction
            print('angle', np.dot(gVec.T,x-self.wopt)/(np.linalg.norm(gVec)*np.linalg.norm(x-self.wopt)))



            iterations_list.append( x.copy() )
            local_gradients_list.append( gVec.copy()  )
            
            iterations = numpy.hstack( iterations_list, dtype=self.dtype_ )
            local_gradients = numpy.hstack( local_gradients_list, dtype=self.dtype_ )

            #print pairwise angle
            norms = np.linalg.norm(local_gradients, axis=0)
            normalized_X = local_gradients / norms

            # Compute the dot product between each pair of normalized columns
            dot_products = np.dot(normalized_X.T, normalized_X)
            print('dot_products\n', dot_products)


            S = numpy.diff( iterations ).astype(self.dtype_) 
            Y = numpy.diff( local_gradients ).astype(self.dtype_) 
            print('S.shape:  ', S.shape)

            print('directional gradient:', np.linalg.norm(Y, axis=0)/np.linalg.norm(S, axis=0))

 
            #solve the unconstrained LS problem
            alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   self.damping *lr* gVec + (S - self.damping *lr *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            x -= pVec
            self.errorPerPicard.append(  np.linalg.norm(x- self.wopt) / self.woptnorm )
            self.path.append( x.copy() )


            print('distance', np.linalg.norm(x - self.wopt) )
            self.Newton_info(x, x_old)
            if alpha_lstsq.size>0:
                self.print_alpha(alpha_lstsq)

        
        return self.w-x




    def Newton_GMRES(self, m):
    #This is a fake Newton-GMRES solver,
    
        self.gradnorm_t.append(   numpy.linalg.norm( self.computeGrad().astype(self.dtype_)  )  )

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
        
        print( bnorm )
        if bnorm  < self.Tol:
            self.stop = True

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



    

    
    







        



        
        