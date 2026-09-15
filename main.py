import numpy as np
import numpy.random as rd

def fct1(x, sigma, mu):
    return np.exp(-((x-mu)**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def algo(x0, sigma0, N):
    tab = [x0]
    for i in range(1, N+1):
        x = rd.normal(tab[-1], sigma0)



