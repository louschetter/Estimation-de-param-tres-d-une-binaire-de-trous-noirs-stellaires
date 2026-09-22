import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import random

#Distributions que l'on ne connaitra pas lorsque qu'on utilisera les données réelles
def gaus1(x, sigmaf=1, mu=0):
    return np.exp(-((x-mu)**2)/(2*sigmaf**2))/np.sqrt(2*np.pi*sigmaf**2)

def gaus2(x, sigmaf=1, diffmu=5):
    mu1 = -diffmu/2
    mu2 = diffmu/2
    return np.exp(-((x-mu1)**2)/(2*sigmaf**2))/np.sqrt(2*np.pi*sigmaf**2) + np.exp(-((x-mu2)**2)/(2*sigmaf**2))/np.sqrt(2*np.pi*sigmaf**2)



#Algo
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



# On fait varier sigmaf, l'écart-type de la fonction f proportionnelle à la distribution cible
x1_1 = algo(0, 1, gaus1, 1, 0, 100000)
x1_8 = algo(0, 1, gaus1, 8, 0, 100000)
x1_17 = algo(0, 1, gaus1, 17, 0, 100000)

plt.figure()
plt.title('Algorithme MH - Gaussienne simple')
plt.xlabel('')
plt.ylabel('')
plt.hist(x1_1, bins=200, density=True,histtype='step', label=r"$\sigma_f$ = 1")
plt.hist(x1_8, bins=200, density=True,histtype='step', label=r"$\sigma_f$ = 8")
plt.hist(x1_17, bins=200, density=True,histtype='step', label=r"$\sigma_f$ = 17")
plt.legend()
plt.savefig('G1Varsigmaf')



# On fait varier sigma0, l'ecart-type de la loi qui suit le point suivant (ici on a fait varié )
x1_1 = algo(0, 1, gaus1, 1, 0, 100000)
x800_1 = algo(0, 800, gaus1, 1, 0, 100000)
x17000_1 = algo(0, 17000, gaus1, 1, 0, 100000)

plt.figure()
plt.title('Algorithme MH - Gaussienne simple')
plt.xlabel('')
plt.ylabel('')
plt.hist(x1_1, bins=200, density=True, histtype='step', label=r"$\sigma_0$ = 1")
plt.hist(x800_1, bins=200, density=True,histtype='step', label=r"$\sigma_0$ = 80")
plt.hist(x17000_1, bins=200, density=True,histtype='step', label=r"$\sigma_0$ = 170")
plt.legend()
plt.savefig('G1varsigma0')


# On essaye avec la double gaussienne, on fait varier sigma0

plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - $\sigma_0$ = 0.5')
plt.xlabel('')
plt.ylabel('')
for i in range(1, 6) : 
    plt.hist(algo(5, 0.5, gaus2, 1, 10, 100000), bins=200, density=True, histtype='step')
#plt.legend()
plt.savefig('G2sigma0_05')
#ca parcourt quasi jamais

plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - $\sigma_0$ = 0.75')
plt.xlabel('')
plt.ylabel('')
for i in range(1, 6) : 
    plt.hist(algo(5, 0.75, gaus2, 1, 10, 100000), bins=200, density=True, histtype='step')
#plt.legend()
plt.savefig('G2sigma0_075')
#ca parcourt rarement


plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - $\sigma_0$ = 1')
plt.xlabel('')
plt.ylabel('')
for i in range(1, 6) : 
    plt.hist(algo(5, 1, gaus2, 1, 10, 100000), bins=200, density=True, histtype='step')
#plt.legend()
plt.savefig('G2sigma0_1')
# ca parcourt desfois mais pas toujours

plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - $\sigma_0$ = 2')
plt.xlabel('')
plt.ylabel('')
for i in range(1, 6) : 
    plt.hist(algo(5, 2, gaus2, 1, 10, 100000), bins=200, density=True, histtype='step')
#plt.legend()
plt.savefig('G2sigma0_2')
#ca parcourt toujours

plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - $\sigma_0$ = 3')
plt.xlabel('')
plt.ylabel('')
for i in range(1, 6) : 
    plt.hist(algo(5, 3, gaus2, 1, 10, 100000), bins=200, density=True, histtype='step')
#plt.legend()
plt.savefig('G2sigma0_3')
#ca parcourt bien les 2 gaussiennes


plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - $\sigma_0$ = 5')
plt.xlabel('')
plt.ylabel('')
for i in range(1, 6) : 
    plt.hist(algo(5, 5, gaus2, 1, 10, 100000), bins=200, density=True, histtype='step')
#plt.legend()
plt.savefig('G2sigma0_5')
#ca parcourt bien les 2 gaussiennes


def algo2(x0, listesigma0, fonction, sigmaf, muf, d, N):
    tab = np.ones([1,d])*x0
    for i in range(1, N+1):
        x = rd.normal(tab[-1, :], random.choice(listesigma0))
        alpha = fonction(x, sigmaf, muf)/fonction(tab[-1,:], sigmaf, muf)
        u = rd.uniform(0, 1, size=[1,d])

        resa = (u < alpha).astype(int)*x
        resb = (u >= alpha).astype(int)*tab[-1]
        res = resa + resb
        tab = np.vstack([tab, res])
    return np.reshape(tab, d*(N+1))

plt.figure()
plt.title(r'Algorithme MH - Gaussienne double - plusieurs explorateurs')
plt.xlabel('')
plt.ylabel('')
plt.hist(algo2(5, [0.5, 4], gaus2, 1, 10, 5, 100000), bins=200, density=True, histtype='step')
plt.savefig('G1')
