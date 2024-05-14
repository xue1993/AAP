import numpy
import numpy as np


def cgSolver_J(J, b,   MaxIter=1000):
    '''
    Solve J * w = b.
    '''
    d = J.shape[1]
    b = b.reshape(d, 1)
    w = numpy.zeros((d, 1))
    r = b - J @ w
    p = r.copy()
    rsold = numpy.sum(r ** 2)

    for i in range(MaxIter):
        Ap = J @ p
        alpha = rsold / numpy.dot(p.T, Ap)
        w += alpha * p
        r -= alpha * Ap
        rsnew = numpy.sum(r ** 2)

        p = r + (rsnew / rsold) * p
        rsold = rsnew    

    return w

def cgSolver(A, b, lam, Tol=1e-16, MaxIter=1000):
    '''
    Solve (A^T * A + lam * I) * w = b.
    '''
    d = A.shape[1]
    b = b.reshape(d, 1)
    tol = Tol * numpy.linalg.norm(b)
    w = numpy.zeros((d, 1))
    r = b - lam * w - numpy.dot(A.T, numpy.dot(A, w))
    p = r
    rsold = numpy.sum(r ** 2)

    for i in range(MaxIter):
        Ap = lam * p + numpy.dot(A.T, numpy.dot(A, p))
        alpha = rsold / numpy.dot(p.T, Ap)
        w += alpha * p
        r -= alpha * Ap
        rsnew = numpy.sum(r ** 2)
        if numpy.sqrt(rsnew) < tol:
            print('Converged! res = ' + str(rsnew))
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew    

    #print('res = ' + str(rsnew) + ',   iter = ' + str(i))
    #if i == MaxIter-1:
        #print('Warn: CG does not converge! Res = '  + str(rsnew))
    return w




    