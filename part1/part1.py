from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex
import numpy as np
import egttools as egt

import matplotlib.pyplot as plt
from egtplot import plot_static


###############Part 1######################################
# 0 -> Defector
# 1 -> Fair-sharer
# 2 -> Altruist

def moran_step(current_state, beta, mu, Z, A):
    '''
	This function implements a birth-death process over 
	the population. At time t, two players are randomly 
	selected from the population.
	'''
    selected = np.random.choice(current_state, size=2) 
    fitness = estimate_fitness(selected.tolist().count(selected[1]), selected[1], selected[0], Z, A)
    avg_payoff = (fitness[0]+fitness[1])/2
    if np.random.rand() < mu:
        current_state[current_state.index(selected[0])] = np.random.choice([0,1,2], size=1)[0]
    elif np.random.rand() < prob_imitation(beta, fitness):
        current_state[current_state.index(selected[0])] = selected[1]
    return current_state,avg_payoff



def estimate_stationary_distribution(nb_runs, transitory, nbgenerations, beta, mu, Z, A):
    '''
	Return the stationary distribution of the population as a vector of floats 
	containing the fraction of time the population spends in each possible state.
	'''
    stationnary_distribution_0 = 0
    stationnary_distribution_1 = 0
    stationnary_distribution_2 = 0
    average_payoff = 0.0
    for nb in range(nb_runs):
        random_population = np.random.choice([0,1,2],replace=True, size=Z).tolist()
        for i in range(transitory):
            random_population,_ = moran_step(random_population, beta, mu, Z, A)
        for j in range(nbgenerations):
            random_population,payoff = moran_step(random_population, beta, mu, Z, A)
            average_payoff += payoff
            stationnary_distribution_0 += random_population.count(0) # count each possible state
            stationnary_distribution_1 += random_population.count(1)
            stationnary_distribution_2 += random_population.count(2)
    stationnary_distribution_0 = (stationnary_distribution_0 / (nbgenerations*nb_runs))/Z
    stationnary_distribution_1 = (stationnary_distribution_1 / (nbgenerations*nb_runs))/Z
    stationnary_distribution_2 = (stationnary_distribution_2 / (nbgenerations*nb_runs))/Z
    average_payoff = average_payoff / (nbgenerations*nb_runs)
    return [stationnary_distribution_0,stationnary_distribution_1,stationnary_distribution_2],average_payoff

def estimate_fitness(k, invader, resident, N, A):
    '''
	The fitness function determines the average payoff of k 
	invaders and N-k residents in the population of N players. 
	'''
    resultA = (((k-1)*A[invader][invader]) +
               ((N-k)*A[invader, resident]))/float(N-1)
    resultB = ((k*A[resident][invader]) +
               ((N-k-1)*A[resident, resident]))/float(N-1)
    return [resultA, resultB]


def prob_imitation(b, fitness):
    '''
	The probability that the first player imitates the second.
	'''
    return 1./(1. + np.exp(-b*(fitness[0]-fitness[1])))



p = 2/3
E = 1

payoffs = [[(1-p)*E],[(1-p)*E],[E],
		   [(1-p)*E/2],[E/2],[E/2],
		   [0],[0],[0]]

type_labels = ['D','F','A'] 
simplex = plot_static(payoffs, vert_labels=type_labels)
simplex.set_size_inches(10,5)
plt.show()

E=1
nb_generations = 7500
nb_runs = 700
Z = 50
mu = 0.0
beta = 1
probs = [i for i in np.arange(0.0,1.1,step=0.1)]
res_avg_payoff = []
res0_r1 = []
res1_r1 = []
res2_r1 = []
for p in probs:
    print("p = {}".format(p))
    payoffs = np.array([[(1-p)*E,(1-p)*E,E],
                        [(1-p)*E/2,E/2,E/2],
                        [0,0,0]])
    esd,avg_payoff = estimate_stationary_distribution(nb_runs,0,nb_generations,beta,mu,Z,payoffs)

    res_avg_payoff.append(avg_payoff)
    res0_r1.append(esd[0])
    res1_r1.append(esd[1])
    res2_r1.append(esd[2])

plt.figure(figsize=(10,5))
plt.plot(probs,res0_r1,label="D")
plt.plot(probs,res1_r1,label="F")
plt.plot(probs,res2_r1,label="A")
plt.xlabel("Risk probability p")
plt.ylabel("Abundance of strategies")
plt.legend()
plt.show()
        