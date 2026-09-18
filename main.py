import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

def fct1(x, sigma=1, mu=0):
    return np.exp(-((x-mu)**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def algo(x0, sigma0, fonction, sigmaf, muf, N):
    tab = [x0]
    for i in range(1, N+1):
        x = rd.normal(tab[-1], sigma0)
        alpha = fonction(x, sigmaf, muf)/fonction(tab[-1], sigmaf, muf)
        u = rd.uniform(0, 1)
        if u < alpha:
            tab.append(x)
        else:
            tab.append(tab[-1])
    return np.array(tab)


#On fait varier sigma0, l'ecart-type de la loi que suit le point suivant
x1_1 = algo(0, 1, fct1, 1, 0, 100000)
x800_1 = algo(0, 800, fct1, 1, 0, 100000)
x17000_1 = algo(0, 17000, fct1, 1, 0, 100000)

#On fait varier sigmaf, l'écart-type de la fonction f proportionnelle ala distribution cible
x1_1 = algo(0, 1, fct1, 1, 0, 100000)
x1_8 = algo(0, 17, fct1, 8, 0, 100000)
x1_17 = algo(0, 17, fct1, 17, 0, 100000)

plt.figure()
plt.title('Algorithme Métropolis-Hastings')
plt.xlabel('')
plt.ylabel('')
plt.hist(x1_1, bins=200, histtype='step', label=r"$\sigma_0$ = 1")
plt.hist(x800_1, bins=200, histtype='step', label=r"$\sigma_0$ = 80")
plt.hist(x17000_1, bins=200, histtype='step', label=r"$\sigma_0$ = 170")
plt.legend()
plt.savefig('HistMH_variationsigma0')

plt.figure()
plt.title('Algorithme Métropolis-Hastings')
plt.xlabel('')
plt.ylabel('')
plt.hist(x1_1, bins=200, histtype='step', label=r"$\sigma_f$ = 1")
plt.hist(x1_8, bins=200, histtype='step', label=r"$\sigma_f$ = 8")
plt.hist(x1_17, bins=200, histtype='step', label=r"$\sigma_f$ = 17")
plt.legend()
plt.savefig('HistMH_variationsigmaf')