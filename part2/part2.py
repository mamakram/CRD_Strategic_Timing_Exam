import numpy as np
import matplotlib.pyplot as plt

def moran_step_1(current_state, beta, mu, Z, A):
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



def estimate_stationary_distribution_1(nb_runs, transitory, nbgenerations, beta, mu, Z, A):
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
            random_population,_ = moran_step_1(random_population, beta, mu, Z, A)
        for j in range(nbgenerations):
            random_population,payoff = moran_step_1(random_population, beta, mu, Z, A)
            average_payoff += payoff
            stationnary_distribution_0 += random_population.count(0) # count each possible state
            stationnary_distribution_1 += random_population.count(1)
            stationnary_distribution_2 += random_population.count(2)
    stationnary_distribution_0 = (stationnary_distribution_0 / (nbgenerations*nb_runs))/Z
    stationnary_distribution_1 = (stationnary_distribution_1 / (nbgenerations*nb_runs))/Z
    stationnary_distribution_2 = (stationnary_distribution_2 / (nbgenerations*nb_runs))/Z
    average_payoff = average_payoff / (nbgenerations*nb_runs)
    return [stationnary_distribution_0,stationnary_distribution_1,stationnary_distribution_2],average_payoff
	
#####################################################
#################### Part 2 #########################

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
        current_state[current_state.index(selected[0])] = np.random.choice([0,1,2,3,4,5,6,7], size=1)[0]
    elif np.random.rand() < prob_imitation(beta, fitness):
        current_state[current_state.index(selected[0])] = selected[1]
    return current_state,avg_payoff


def estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, A):
    '''
	Return the stationary distribution of the population as a vector of floats 
	containing the fraction of time the population spends in each possible state.
    Z is the size of the population. 
	'''
    stationnary_distribution = [0 for i in range(8)]
    contrib_round_1 = 0.0
    contrib_round_2 = 0.0
    average_payoff = 0.0
    for nb in range(nb_generations):
        #print(nb)
        random_population = np.random.choice([0,1,2,3,4,5,6,7],replace=True, size=Z).tolist()
        for i in range(transitory):
            random_population,_ = moran_step(random_population, beta, mu, Z, A)
        for j in range(nb_runs):
            random_population,payoff = moran_step(random_population, beta, mu, Z, A)
            average_payoff += payoff
            for i in range(8):
                stationnary_distribution[i] += random_population.count(i) # count each possible state
    for i in range(8):
                (stationnary_distribution[i]/(nb_generations*nb_runs))/Z # count each possible state
    contrib_round_1 = stationnary_distribution[3] #+ stationnary_distribution_5 + stationnary_distribution_6 + stationnary_distribution_7)
    contrib_round_2 = (stationnary_distribution[1] + stationnary_distribution[3]) #+ stationnary_distribution_5 + stationnary_distribution_6 + stationnary_distribution_7)
    tot_contrib = contrib_round_1 + contrib_round_2
    average_payoff = average_payoff / (nb_generations*nb_runs)
    return stationnary_distribution,contrib_round_1,contrib_round_2,tot_contrib,average_payoff
	

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



E = 1
res = [[] for i in range(8)]
probs = [i for i in np.arange(0.0,1.1,step=0.1)]
res_r1 = []
res_r2 = []
res_r1r2 = []
res_avg_payoff = []
res_avg_payoff_2 = []
nb_generations = 7500
nb_runs = 700
Z = 50
mu = 0.0
beta = 1
for p in probs:
    print("p= {}".format(p))
    payoffs = np.array([[(1-p)*E,(1-p)*E,       (1-p)*E,        (1-p)*E,        (1-p)*E,    E,      (1-p)*E,    E],
            [(1-p)*E/2,      E/2,           (1-p)*E/2,      (1-p)*E/2,      (1-p)*E,    E,      (1-p)*E,    E],
            [(1-p)*E,        (1-p)*E,       (1-p)*E,        (1-p)*E,        E/2,        E/2,    E/2,        E],
            [(1-p)*E/2,      E/2,           (1-p)*E/2,      E/2,            E/2,        E/2,    E/2,        E/2],
            [(1-p)*E/2,      (1-p)*E/2,     E/2,            E/2,            E/2,        E/2,    E/2,        E/2],
            [0,              0,             0,              0,              E/2,        E/2,    E/2,        E/2],
            [(1-p)*E/2,      (1-p)*E/2,     E/2,            E/2,            0,          0,      0,          0],
            [0,              0,             0,              0,              0,          0,      0,          0]])
    esd,r1,r2,r1r2,avg_payoff = estimate_stationary_distribution(nb_runs,0,nb_generations,beta,mu,Z,payoffs) #Meilleur para : 250,0,7000,1,0,50
    payoffs = np.array([[(1-p)*E,(1-p)*E,E],
                        [(1-p)*E/2,E/2,E/2],
                        [0,0,0]])
    esd_2,avg_payoff_2 = estimate_stationary_distribution_1(nb_runs,0,nb_generations,beta,mu,Z,payoffs)
    for i in range(8):
        res[i].append(esd[i])
    res_r1.append(r1)
    res_r2.append(r2)
    res_r1r2.append(r1r2)
    res_avg_payoff.append(avg_payoff)
    res_avg_payoff_2.append(avg_payoff_2)

labels = ['(0,0,0)','(0,0,1)','(0,1,0)','(0,1,1)','(1,0,0)','(1,0,1)','(1,1,0)','(1,1,1)']
for i in range(8):
    plt.plot(probs,res[i], label=labels[i])
plt.xlabel("Risk probability p")
plt.ylabel("Abundance of strategies")
plt.legend(labels, loc="lower right")
plt.show()

plt.plot(probs,res_r1,label="round1")
plt.plot(probs,res_r2,label="round2")
plt.plot(probs,res_r1r2,label="total")
plt.xlabel("Risk probability p")
plt.ylabel("Probability of contributions")
plt.legend()
plt.show()

plt.plot(probs,res_avg_payoff_2,label="one round game",color='red')
plt.plot(probs,res_avg_payoff,label="two round game",color='b')
max_payoff = res_avg_payoff[:5] + [0.5,0.5,0.5,0.5,0.5,0.5]
plt.fill_between(probs, 0, max_payoff, color='grey', alpha=0.2)
plt.xlabel("Risk probability p")
plt.ylabel("Average payoff")
plt.axis([0,1.0, 0,1.0])
plt.legend()
plt.show()
        