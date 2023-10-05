import numpy as np
import pandas as pd
import random
import operator

# METODOS/FUNÇÕES

# Metodo para criar/startar a população
def create_starting_population(size,Number_of_city):
    population = []
    
    for i in range(0,size):
        population.append(create_new_member(Number_of_city))
        
    return population

# Roleta para escolha dos companheiros
def pick_mate(NumberCitie):
    i = random.randint(0,NumberCitie)    
    return i

# Calcula a distância entre duas cidades
def distance(x,y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

# Calcula a pontuação de toda a população
def score_population(population, CityList):
    scores = []
  
    for i in population:
        scores.append(fitness(i, CityList))
    return scores

# A aptidão(fitness) individual das rotas é calculada aqui
def fitness(route,CityList):
    score = 0
    for i in range(1,len(route)):
        k = int(route[i-1])
        l = int(route[i])

        score = score + distance(CityList[k],CityList[l])
  
    return score

# Criando novo membro da população
def create_new_member(Number_of_city):
    pop = set(np.arange(Number_of_city,dtype=int))
    route = list(random.sample(pop,Number_of_city))
            
    return route

# Cruzamento
def crossover(a,b):
    child = []
    childA = []
    childB = []
    
    geneA = int(random.random()* len(a))
    geneB = int(random.random()* len(a))
    
    start_gene = min(geneA,geneB)
    end_gene = max(geneA,geneB)
    
    for i in range(start_gene,end_gene):
        childA.append(a[i])
        
    childB = [item for item in a if item not in childA]
    child = childA + childB

    return child

# Mutação
def mutate(route,probablity):

    route=np.array(route)
    for swaping_p in range(len(route)):
        if(random.random() < probablity):
            swapedWith = np.random.randint(0,len(route))
            
            temp1=route[swaping_p]
            
            temp2=route[swapedWith]
            route[swapedWith] = temp1
            route[swaping_p] = temp2
    
    return route


def selection(popRanked, eliteSize):
    selectionResults = []
    result=[]
    for i in popRanked:
        result.append(i[0])
    for i in range(0,eliteSize):
        selectionResults.append(result[i])
    
    return selectionResults

def rankRoutes(population,City_List):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = fitness(population[i],City_List)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)

def breedPopulation(mating_pool):
    children = []
    for i in range(len(mating_pool)-1):
            children.append(crossover(mating_pool[i],mating_pool[i+1]))
    return children

def mutatePopulation(children,mutation_rate):
    new_generation = []
    for i in children:
        muated_child=mutate(i,mutation_rate)
        new_generation.append(muated_child)
    return new_generation

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def next_generation(City_List, current_population, mutation_rate, elite_size):
    population_rank = rankRoutes(current_population, City_List)
    selection_result = selection(population_rank, elite_size)
    mating_pool = matingPool(current_population, selection_result)
    children = breedPopulation(mating_pool)
    next_generation = mutatePopulation(children, mutation_rate)

    return next_generation

def genetic_algorithm(City_List, size_population, elite_size, mutation_Rate, generation):
    pop = []
    progress = []
    
    Number_of_cities=len(City_List)
    
    population = create_starting_population(size_population,Number_of_cities)
    progress.append(rankRoutes(population,City_List)[0][1])
    print(f"Distacia da Rota Inicial {progress[0]}")
    print(f"Rota Inicial {population[0]}")
    for i in range(0, generation):
        pop = next_generation(City_List,population,mutation_Rate,elite_size)
        progress.append(rankRoutes(pop,City_List)[0][1])
  
    rank_ = rankRoutes(pop,City_List)[0]

    return rank_, pop

##############################################################################
# FIM DOS METODOS/FUNÇÕES
##############################################################################

# Altera aqui os valores
Num_de_cidades = 25

Tamanho_da_populacao = 1000
Number_best_route = 100
Taxa_de_Mutacao = 0.01 # equivalente a 1%, indo de [0,1] onde 1 é 100%
Num_de_Geracao = 2000

cityList = [(113, 188), (55, 131), (145, 197), (98, 107), (106, 31), 
            (43, 180), (48, 67), (125, 189), (95, 55), (56, 171), 
            (37, 194), (1, 94), (105, 101), (164, 18), (34, 102), 
            (37, 178), (55, 160), (84, 60), (178, 28), (8, 33), 
            (111, 153), (11, 14), (162, 136), (93, 8), (52, 122)]

'''  
# Gerar aleatoriamente a distancia entre as cidades         
for i in range(0, Num_de_cidades):
    x=int(random.random() * 200)
    y=int(random.random() * 200)
    cityList.append((x,y))
'''

rank_,pop = genetic_algorithm(City_List=cityList, 
                            size_population=Tamanho_da_populacao, 
                            elite_size=Number_best_route, 
                            mutation_Rate=Taxa_de_Mutacao, 
                            generation=Num_de_Geracao
                            )

print(f"Melhor Rota :{pop[rank_[0]]} ")
print(f"Melhor Distancia da Rota {rank_[1]}")
