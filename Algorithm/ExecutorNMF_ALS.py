import numpy
import numpy as np


class Executor:
    def __init__(self, A, k, W, H, dtype_=np.double):
        
        self.dtype_ = dtype_
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.k = k 
        self.Wdim = self.m * self.k
        
        self.A = A.copy().astype(self.dtype_)
        self.W = W.copy()  
        self.W[self.W < 0] = 0
        self.H = H.copy() 

        #update direction
        self.pW = np.zeros((self.m, self.k), dtype=self.dtype_) 
        self.pH = np.zeros((self.k, self.n), dtype=self.dtype_)
        self.iterations_history = []
        self.residuals_history = []
        self.stop = False

        #Default Params of algo
        self.warmup_ = False
        self.damping = 1

        #Error tractor
        self.funcVals = []        
        self.funcVals.append( self.objFun( self.W, self.H) )





    
    #take 5 ALS step as the warm up
    def warmUp(self):
        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
     

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(5):
            self.iterations_history.append(  self.flatten_(W,H).copy() )

            W,H = self.oneALS_(W.copy(),H.copy())
            self.funcVals.append( self.objFun( W,H) )
            print( self.funcVals[-1] )             
             
            self.residuals_history.append(   self.flatten_( W,H)  - self.iterations_history[-1]  )
            
            
        self.W = W 
        self.H = H

        

    def setParam(self, a, isSearch,  etaList, damping_=1, warmup_=False):   

        self.damping = damping_ #damping ratio
        self.isSearch = isSearch
        self.etaList = np.array(etaList, dtype=self.dtype_)
        self.numEta = len(self.etaList)
        self.warmup_ = warmup_
        
        if self.warmup_:
            self.warmUp()
        


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
    def Picard( self, lr =None, local_epochs = 5 ):     


        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
     

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            W,H = self.oneALS_(W.copy(),H.copy())

            self.funcVals.append( self.objFun( W,H) )
            print( self.funcVals[-1] )
            
        
        return self.W-W, self.H-H


    #AAP method
    def AAP( self,  lr=None, local_epochs = 5 ):     


        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
    

        WH_list = []
        residuals_list = []

        #take gd steps, since the last update is is not used, local_epoch plus one
        for i in range(local_epochs+1):
            WH_list.append(  self.flatten_(W,H).copy() )

            W,H = self.oneALS_(W.copy(),H.copy())
            self.funcVals.append( self.objFun( W,H) )
            print( self.funcVals[-1] )

            residuals_list.append(  self.flatten_( W,H )  - WH_list[-1] )



        iterations = numpy.hstack( WH_list, dtype=self.dtype_ )
        residuals = numpy.hstack( residuals_list, dtype=self.dtype_ )

        gVec = residuals_list[0]
        S = numpy.diff( iterations ).astype(self.dtype_) 
        Y = numpy.diff( residuals ).astype(self.dtype_) 
        print('S.shape:  ', S.shape)
 
        #solve the unconstrained LS problem
        alpha_lstsq = np.linalg.lstsq(Y[:self.Wdim,:].astype(np.float64), gVec[:self.Wdim,:].astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
      
        #Update AAP direction with alpha_lstsq solutin
        pVec =   -self.damping *gVec + (S + self.damping *Y) @ alpha_lstsq 
        print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )
        pW = pVec[:(self.m*self.k)].reshape( self.m, self.k)
        pH = pVec[(self.m*self.k):].reshape( self.k, self.n)

        #MODIFY TEH LAST FUNCTION VALUES
        W = self.W-pW
        W[W < 0] = 0
        H = self.H-pH
        H[H < 0] = 0
        self.funcVals[-1] = self.objFun( W, H) 
        print( self.funcVals)
        
        return self.W-W, self.H-H

    
    #Classical AA(m) method
    def AA( self,  lr=None, m = 5 ):     
        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_)         


        for i in range(m+1):

            self.iterations_history.append(  self.flatten_(W,H).copy() )

            print( self.objFun( W, H ) )
            W_,H_ = self.oneALS_(W.copy(),H.copy())
            print( 'AFter: ', self.objFun( W_,H_ ) )
             
            self.residuals_history.append(   self.flatten_( W_,H_ )  - self.iterations_history[-1]  )

            while len( self.iterations_history )> (m+1):
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
            alpha_lstsq = np.linalg.lstsq(Y[:self.Wdim,:].astype(np.float64), gVec[:self.Wdim,:].astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   -self.damping  *gVec + (S + self.damping *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            
            W -= pVec[:(self.m*self.k)].reshape( self.m, self.k)
            W[W < 0] = 0
            H -= pVec[(self.m*self.k):].reshape( self.k, self.n)
            H[H < 0] = 0
            

            self.funcVals.append( self.objFun( W,H) )
            print( self.funcVals[-1] )
        
        return self.W-W, self.H-H


    #resAA method
    def resAA( self,  lr=None, m = 5 ):     


        #intialization
        W = self.W.copy().astype(self.dtype_) 
        H = self.H.copy().astype(self.dtype_) 
    

        WH_list = []
        residuals_list = []        

        #take gd steps
        for i in range(m+1):
            WH_list.append(  self.flatten_(W,H).copy() )


            print( self.objFun( W, H ) )
            W_,H_ = self.oneALS_(W.copy(),H.copy())
            print( 'AFter: ', self.objFun( W_,H_ ) )

            residuals_list.append(  self.flatten_( W_,H_ )  - WH_list[-1] )
            
            iterations = numpy.hstack( WH_list, dtype=self.dtype_ )
            residuals = numpy.hstack( residuals_list, dtype=self.dtype_ )

            gVec = residuals_list[-1]
            S = numpy.diff( iterations ).astype(self.dtype_) 
            Y = numpy.diff( residuals ).astype(self.dtype_) 
            print('S.shape:  ', S.shape)
 
            #solve the unconstrained LS problem
            alpha_lstsq = np.linalg.lstsq(Y[:self.Wdim,:].astype(np.float64), gVec[:self.Wdim,:].astype(np.float64),rcond=-1)[0].astype(self.dtype_)      
        
            #Update AAP direction with alpha_lstsq solutin
            pVec =   -self.damping* gVec + (S + self.damping *Y) @ alpha_lstsq 
            print( 'Length of ptilde:',  numpy.linalg.norm( S @ alpha_lstsq), 'rtilde', numpy.linalg.norm( Y @ alpha_lstsq - gVec  )  )

            W -= pVec[:(self.m*self.k)].reshape( self.m, self.k)
            W[W < 0] = 0
            H -= pVec[(self.m*self.k):].reshape( self.k, self.n)
            H[H < 0] = 0

            self.funcVals.append( self.objFun( W,H) )
            print( self.funcVals[-1] )
        
        return self.W-W, self.H-H




    

    
    







        



        
        