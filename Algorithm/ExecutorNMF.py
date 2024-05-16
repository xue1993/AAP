import numpy
import numpy as np


class Executor:
    def __init__(self, A, k, W=None, H=None, dtype_=np.double):
        
        self.dtype_ = dtype_
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.k = k 
        
        self.A = A.astype(self.dtype_)
        self.W = W    
        self.H = H 

        #update direction
        self.pW = np.zeros((self.m, self.k), dtype=self.dtype_) 
        self.pH = np.zeros((self.k, self.n), dtype=self.dtype_)
        self.iterations_history = []
        self.residuals_history = []
        self.stop = False

        

    def setParam(self, a, isSearch,  etaList):   

        self.damping = 1 #damping ratio
        self.isSearch = isSearch
        self.etaList = np.array(etaList, dtype=self.dtype_)
        self.numEta = len(self.etaList)
        


    def updateP(self, pW,pH):
        self.pW = pW.copy().astype(self.dtype_)
        self.pH = pH.copy().astype(self.dtype_)
        

    def updateWH(self):
        self.W -= self.pW
        self.H -= self.pH

    def updateWH_(self, W, H):
        self.W =W.copy().astype(self.dtype_)
        self.H =H.copy().astype(self.dtype_)
      
    
    def objFun(self, W,H):
        '''
        relative Frobenius norm of the decomposition
        f (w) = ||A-W@H||_F/||A||_F 
        '''
        return np.linalg.norm(self.A -W @ H,'fro') / np.linalg.norm(self.A, 'fro')
    
    def objFunSearch(self):
        objValVec = np.zeros(self.numEta + 1, dtype=self.dtype_)
        for l in range(self.numEta):
            objValVec[l] = self.objFun( self.W - self.etaList[l] * self.pW, self.H - self.etaList[l] * self.pH )
        objValVec[-1] = self.objFun(self.W , self.H)
        return objValVec
    
    def oneALS(self):

        W = self.W.copy()

        # Normalize W 
        norm2 = np.sqrt(np.sum(W**2, axis=0))
        toNormalize = norm2 > 0
        if np.any(toNormalize):
            W[:, toNormalize] = W[:, toNormalize] / norm2[toNormalize]

        # Update H
        H = np.linalg.lstsq(W, self.A, rcond=None)[0]
        H[H < 0] = 0

        # Update W
        W = np.linalg.lstsq(H.T, self.A.T, rcond=None)[0].T
        W[W < 0] = 0

        return W, H

    def oneALS_(self,W,H):

        # Normalize W 
        norm2 = np.sqrt(np.sum(W**2, axis=0))
        toNormalize = norm2 > 0
        if np.any(toNormalize):
            W[:, toNormalize] = W[:, toNormalize] / norm2[toNormalize]

        # Update H
        H = np.linalg.lstsq(W, self.A, rcond=None)[0]
        H[H < 0] = 0

        # Update W
        W = np.linalg.lstsq(H.T, self.A.T, rcond=None)[0].T
        W[W < 0] = 0        

        return W, H
    
    def flatten_(self,W,H):
        return (np.concatenate([W.flatten(), H.flatten()])).reshape(-1,1)

    #Picard, multiple ALS steps
    def Picard( self, lr, local_epochs = 5 ):     


        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
     

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            W,H = self.oneALS_(W.copy(),H.copy())
            print( self.objFun( W,H) )
            
        
        return self.W-W, self.H-H


    #AAP method
    def AAP( self,  lr=1, local_epochs = 5 ):     


        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
    

        WH_list = []
        residuals_list = []

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            WH_list.append(  self.flatten_(W,H).copy() )

            W,H = self.oneALS_(W.copy(),H.copy())
            print( self.objFun(W,H) )

            residuals_list.append(  self.flatten_( W,H )  - WH_list[-1] )



        iterations = numpy.hstack( WH_list, dtype=self.dtype_ )
        residuals = numpy.hstack( residuals_list, dtype=self.dtype_ )

        gVec = residuals_list[0]
        S = numpy.diff( iterations ).astype(self.dtype_) 
        Y = numpy.diff( residuals ).astype(self.dtype_) 
        print('S.shape:  ', S.shape)
 
        #solve the unconstrained LS problem
        alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
      
        #Update AAP direction with alpha_lstsq solutin
        pVec =   self.damping *lr* gVec + (S - self.damping *lr* Y) @ alpha_lstsq 
        print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )
        
        return pVec[:(self.m*self.k)].reshape( self.m, self.k), pVec[(self.m*self.k):].reshape( self.k, self.n)

    
    #Classical AA(m) method
    def AA( self,  lr=1, m = 5 ):     
        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_)         


        for i in range(m+1):

            self.iterations_history.append(  self.flatten_(W,H).copy() )
             
            print( self.objFun( W,H) )
             
            self.residuals_history.append(   self.flatten_( *self.oneALS_(W.copy(),H.copy()) )  - self.iterations_history[-1]  )

            if len( self.iterations_history )> (m+1):
                self.iterations_history.pop(0)
                self.residuals_history.pop(0)               

                
            #Update S Y matrix
            iterations = numpy.hstack( self.iterations_history, dtype=self.dtype_ )
            residuals = numpy.hstack( self.residuals_history, dtype=self.dtype_ )

            gVec = self.residuals_history[-1].copy()
            S = numpy.diff( iterations ).astype(self.dtype_) 
            Y = numpy.diff( residuals ).astype(self.dtype_) 
            print('S.shape:  ', S.shape)
    
            #solve the unconstrained LS problem
            alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   self.damping * lr *gVec + (S - self.damping *lr *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            
            W -= pVec[:(self.m*self.k)].reshape( self.m, self.k)
            H -= pVec[(self.m*self.k):].reshape( self.k, self.n)
        
        return self.W-W, self.H-H


    #resAA method
    def resAA( self,  lr=1, m = 5 ):     


        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
    

        WH_list = []
        residuals_list = []        

        #take gd steps
        for i in range(m+1):
            WH_list.append(  self.flatten_(W,H).copy() )

            print( self.objFun( W,H) )

            residuals_list.append(  self.flatten_( *self.oneALS_(W.copy(),H.copy()) )  - WH_list[-1] )
            
            iterations = numpy.hstack( WH_list, dtype=self.dtype_ )
            residuals = numpy.hstack( residuals_list, dtype=self.dtype_ )

            gVec = residuals_list[-1]
            S = numpy.diff( iterations ).astype(self.dtype_) 
            Y = numpy.diff( residuals ).astype(self.dtype_) 
            print('S.shape:  ', S.shape)
 
            #solve the unconstrained LS problem
            alpha_lstsq = np.linalg.lstsq(Y.astype(np.float64), gVec.astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   self.damping *lr* gVec + (S - self.damping *lr *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            W -= pVec[:(self.m*self.k)].reshape( self.m, self.k)
            H -= pVec[(self.m*self.k):].reshape( self.k, self.n)
        
        return self.W-W, self.H-H




    

    
    







        



        
        