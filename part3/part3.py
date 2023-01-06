import numpy as np
import egttools as egt
#import egttools.numerical.numerical.
from egttools.games import Matrix2PlayerGameHolder,MatrixNPlayerGameHolder
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline



"""
Each player has a personal strategy [threshold,value_if_over,value_if_under]

"""
def estimate_stationary_distribution(nb_rounds,nb_generations,nb_runs,nb_players,E,p,beta,mu,Z):
    print("run for p = {}, {} rounds, {} players,{},{}".format(p,nb_rounds,nb_players,nb_generations,nb_runs))
    total_payoffs = np.zeros(Z)
    target = E*nb_players/2
    mu = 0.03
    sigma= 0.15
    max_contribution = E/nb_rounds
    strategies = [[np.random.uniform(target),np.random.choice((0,max_contribution)),np.random.choice((0,max_contribution))] for _ in range(Z)]
    contributions_per_round_average = np.zeros(nb_rounds)
    for k in range(nb_generations):
        payoffs = np.zeros(Z)
        pop_counter = [0 for _ in range(Z)]
        contributions_per_round = np.zeros(nb_rounds)
        for _ in range(nb_runs):
            group_payoffs = {i: E for i in np.random.choice(Z,size=nb_players,replace=False)} #E/R par round
            common_pool = 0
            contribution_pool = 0.0
            for i in range(nb_rounds):
                current_pool = common_pool 
                for player in group_payoffs.keys():
                    player_threshold = strategies[player][0]
                    contribution = 0
                    if player_threshold <= current_pool: #contribute common pool faut separer
                        contribution = strategies[player][1]
                    else:
                        contribution = strategies[player][2]
                    contribution_pool += contribution
                    common_pool+= contribution
                    group_payoffs[player] -= contribution
                if common_pool>= target:
                    break
                contributions_per_round[i] += (contribution_pool-contributions_per_round[i])/(nb_runs+1)
            for player in group_payoffs.keys():
                pop_counter[player]+=1
                if common_pool<target:
                    group_payoffs[player] *= (1-p)
                payoffs[player] +=  ((group_payoffs[player]/E)-payoffs[player])/pop_counter[player] #add to average
            
            

        #Evolution de la population
        fitnesses = np.exp(beta*payoffs) #fonction de l'article
        proportions = fitnesses/sum(fitnesses)
        new_pop = np.random.choice(Z,size=Z,p=proportions) #nouvelle population choisie en fct des résultats de l'ancienne population

        new_strategies = []
        threshold_mean = np.mean([s[0] for s in strategies])
        for j in new_pop:
            if np.random.rand() <mu: #mutation
                new_strategies.append([np.random.normal(threshold_mean,sigma),np.random.choice((0,max_contribution)),np.random.choice((0,max_contribution))])
            else:
                new_strategies.append(strategies[j])
        strategies = new_strategies
        total_payoffs+=(payoffs-total_payoffs)/(k+1)
        contributions_per_round_average += (contributions_per_round-contributions_per_round_average)/(k+1)
    return np.mean(total_payoffs), np.divide(np.divide(contributions_per_round,nb_generations*nb_runs),Z)



E=60
nb_players = 6
sigma = 0.15
beta = 1
mu = 0.03
nb_rounds = 6
nb_generations = 1000
nb_runs = 1000

#labels = ['(0;0;0)','(0;0,1)','(0;1,0)','(0;1,1)','(1;0,0)','(1;0,1)','(1;1,0)','(1;1,1)']
round_labels = np.arange(2,14,2)
player_labels = np.arange(2,14,2)
probs = [i for i in np.arange(0.0,1.1,step=0.1)]
final_results = [[] for _ in range(len(round_labels))]
final_results2 = [[] for _ in range(len(player_labels))]
contributions=[]
for p in probs:
    results=[]
    results2=[]
    for rounds in round_labels:
        res,contribution = estimate_stationary_distribution(rounds,nb_generations,nb_runs,nb_players,E,p,beta,mu,100)
        results.append(res)
        if p==0.7:
            contributions.append(contribution)
    for players in player_labels:
        res,_ = estimate_stationary_distribution(nb_rounds,nb_generations,nb_runs,players,E,p,beta,mu,100)
        results2.append(res)
    print("for p={}: ".format(p),results)
    for i in range(len(round_labels)):
        final_results[i].append(results[i])
        final_results2[i].append(results2[i])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Strategy distribution based on risk factor and \nnb rounds (beta={}, sigma={}, mu={})".format(beta,sigma,mu))
plt.xticks(probs)
for i in range(len(round_labels)):
    plt.plot(probs,final_results[i],label=round_labels[i],marker='o')#,linewidth=0.5)
plt.legend(round_labels, loc="upper right")
plt.ylim(bottom=0.0,top=1.0)
plt.ylabel("Average payoffs")
plt.xlabel("Risk probability")
max_payoff = [1.0,0.9,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5]
plt.fill_between(probs, 0, max_payoff, color='grey', alpha=0.2)

plt.subplot(1,2,2)
plt.title("Strategy distribution based on risk factor and \nnb players (beta={}, sigma={}, mu={})".format(beta,sigma,mu))
plt.xticks(probs)
for i in range(len(player_labels)):
    plt.plot(probs,final_results2[i],label=round_labels[i],marker='o')#,linewidth=0.5)
plt.legend(round_labels, loc="upper right")
plt.ylim(bottom=0.0,top=1.0)
plt.ylabel("Average payoffs")
plt.xlabel("Risk probability")
plt.fill_between(probs, 0, max_payoff, color='grey', alpha=0.2)
plt.show()

plt.title("Timing of contributions per round (beta={}, sigma={}, mu={})".format(beta,sigma,mu))
plt.ylabel("Average contributions")
plt.xlabel("Round number")
x = np.arange(0,30)
plt.bar(x[:2], contributions[0], label = "2 rounds")
plt.bar(x[3:9], contributions[2],label = "6 rounds")
plt.bar(x[10:22], contributions[5],label = "12 rounds")
plt.legend(['2 rounds','6 rounds','12 rounds'],loc="upper right")
plt.show()