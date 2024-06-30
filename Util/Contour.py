import numpy as np
from scipy import optimize
import numpy
import time
import matplotlib.pyplot as plt
import sys
import copy
import Util.Logistic as Logistic
from Algorithm.Solver_LQ_plot import Solver

class Contour:
  def __init__(self, xMat, yVec, Gamma, dtype_=np.double):

        self.dtype_ = dtype_
        self.s, self.d = xMat.shape
        self.yVec = yVec.reshape(self.s, 1).astype(self.dtype_)
        self.xMat = (xMat * self.yVec).astype(self.dtype_) #thus the xMat saves the multiplication between data and label
        self.w = np.zeros((self.d, 1), dtype=self.dtype_)
        self.paths = []

        self.gamma = Gamma

        self.fit( xMat, yVec )



  #fit is a function to compute the global minimizer
  def fit(self, xMat, yVec):

        #compute the global minimizer
        solver = Logistic.Solver(X=xMat, y=yVec)
        self.wopt, self.condnum = solver.newton(self.gamma)

        print('the condition number is ',self.condnum)
        print('current xMat shape', self.xMat.shape)



  def Compute_levelset(self,u1,u2,v1,v2):


        #self define two directions and normalize
        self.d1 = self.wopt - self.w
        #there are different choice for d2 direction,
        #self.d2 = self.computeGrad()
        self.d2 = np.random.randn(*self.d1.shape)
        self.d1 = self.d1 / np.linalg.norm(self.d1)
        self.d2 = self.d2 / np.linalg.norm(self.d2)

        #contour
        # Create a grid for visualization
        grid_size = 15
        u = np.linspace(u1, u2, grid_size)
        v = np.linspace(v1, v2, grid_size)
        self.u, self.v = np.meshgrid(u, v)
        self.loss_values = np.zeros_like(self.u)

        # Calculate the loss values over the grid
        for i in range(grid_size):
            for j in range(grid_size):
                params = self.u[i, j] * self.d1 + self.v[i, j] * self.d2
                self.loss_values[i, j] = self.objFun(params)

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



def loadData(filename):

    npzfile = numpy.load(filename)
    print(npzfile.files)
    X = npzfile['X']
    y = npzfile['y']
    n, d = X.shape
    #X = numpy.concatenate((X, numpy.ones((n, 1))), axis=1)
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    return X, y

#run different algorithms and visualize the optimization path

def plot_all(gamma, m, picard=None, AA=None, AAP=None, resAA=None, NewtonCG=None, NewtonGMRES=None):

    if picard is not None:
        plt.semilogy(picard, label='Picard', color='tab:orange', linestyle='--', linewidth=1.5)
    if AA is not None:
        plt.semilogy(AA, label='AA', color='tab:blue', linestyle='--', linewidth=1.5)
    if resAA is not None:
        plt.semilogy(resAA, label='resAA', color='tab:green', linestyle='-.', linewidth=1.5)
    if NewtonGMRES is not None:
        plt.semilogy(NewtonGMRES, label='Newton-GMRES', color='tab:purple', linestyle=':', linewidth=1.5)
    if AAP is not None:
        plt.semilogy(AAP, label='AAP', color='tab:red', linestyle='-', marker='o', markevery=0.1, linewidth=1.5, markersize=5)

    plt.legend()
    plt.title(f'logistic regression with $\eta_0$={gamma}, m={m}$')
    plt.xlabel('global iteration $t$')  # Updated x-axis label
    plt.ylabel('$||w^t - w^*||$')  # Updated y-axis label
    plt.tight_layout()
    plt.show()

class Demo:
    def __init__(self, maxiter, repeat, gamma, wopt):
        self.maxiter = maxiter
        self.repeat = repeat
        self.gamma = gamma
        self.wopt = wopt

    def fit(self, xMat, yVec):
        n, self.d = xMat.shape
        self.xMat = xMat[0:n, :].astype(np.float64)
        self.yVec = yVec[0:n].reshape(n, 1).astype(np.float64)

    def testConvergence(self):
        errMat = numpy.zeros((self.repeat*5, self.maxiter+1))
        cur = 0
        self.paths = []
        for r in range(self.repeat):
            print(str(r) + '-th repeat  ')


            m_ = 5
            print('+++++++   AAP   +++++++')
            solver = Solver(local_epochs = m_, eta0=1, C=0, algo_='AAP')
            solver.fit(self.xMat, self.yVec)
            start_time = time.time()
            err = solver.train(self.gamma, self.wopt, maxIter=self.maxiter)
            print(cur, 'time used:   ' ,time.time() - start_time)
            print('error', err)
            print()
            # Plot the optimization path
            path = np.squeeze(solver.executor.path)
            self.paths.append( path  )
            del solver
            l = min(len(err), self.maxiter+1)
            errMat[cur, 0:l] = err[0:l]
            cur = cur+1


            print('+++++++   AA   +++++++')
            solver = Solver(local_epochs = m_, eta0=1, C=0, algo_='AA')
            solver.fit(self.xMat, self.yVec)
            start_time = time.time()
            err = solver.train(self.gamma, self.wopt, maxIter=self.maxiter)
            print(cur, 'time used:   ' ,time.time() - start_time)
            print('error', err)
            print()
            # Plot the optimization path
            path = np.squeeze(solver.executor.path)
            self.paths.append( path  )
            del solver
            l = min(len(err), self.maxiter+1)
            errMat[cur, 0:l] = err[0:l]
            cur = cur+1

            print('+++++++   resAA   +++++++')
            solver = Solver(local_epochs = m_, eta0=1, C=0, algo_='resAA')
            solver.fit(self.xMat, self.yVec)
            start_time = time.time()
            err = solver.train(self.gamma, self.wopt, maxIter=self.maxiter)
            print(cur, 'time used:   ' ,time.time() - start_time)
            print('error', err)
            print()
            # Plot the optimization path
            path = np.squeeze(solver.executor.path)
            self.paths.append( path  )
            del solver
            l = min(len(err), self.maxiter+1)
            errMat[cur, 0:l] = err[0:l]
            cur = cur+1



            plot_all( self.gamma, m_ , AA=errMat[1], AAP=errMat[0], resAA=errMat[2] )


        return errMat

from matplotlib.colors import Normalize
def visualize_contours(Contour_):
  norm = Normalize(vmin=0.9*np.min(Contour_.loss_values), vmax=np.max(Contour_.loss_values))  # Adjust normalization

  contour = plt.contour(Contour_.u, Contour_.v, Contour_.loss_values, levels=20, cmap='Greys', norm=norm)
  #contour = plt.contour(Contour_.u, Contour_.v, Contour_.loss_values, levels=20, cmap='Greys')
  plt.colorbar(contour, label='Loss Value')



  path = Contour_.paths[1]
  path_u = np.dot(path , Contour_.d1)
  path_v = np.dot(path , Contour_.d2)
  plt.plot(path_u, path_v, label = 'AA($m$)', color='#1f77b4',  marker='o',  markerfacecolor='none', linestyle='-')

  path = Contour_.paths[2]
  path_u = np.dot(path , Contour_.d1)
  path_v = np.dot(path , Contour_.d2)
  plt.plot(path_u, path_v, label = 'resAA($m$)', color='#2ca02c', marker='+', linestyle='-')

  path = Contour_.paths[0]
  path_u = np.dot(path , Contour_.d1)
  path_v = np.dot(path , Contour_.d2)
  plt.plot(path_u, path_v, label = 'AAP($m$)', color='tab:red', marker='.', linestyle='-')

  #plt.title('Loss Landscape and Optimization Path',  fontsize=12)
  plt.xlabel('Direction 1: $w^*-w_0$',  fontsize=11)
  plt.ylabel('Direction 2: random',  fontsize=11)
  plt.scatter(0, 0, color='#FFFF00', marker='*', s=500, label='Start Point',alpha=1)
  plt.legend( fontsize=8)
  plt.show()
