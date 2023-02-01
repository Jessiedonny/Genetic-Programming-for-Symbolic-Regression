
import operator
import math
import random
import pandas as pd
import numpy
from deap import base,creator,tools,gp

#read and prepare data - ds[0] will be part of the terminal set
df= pd.read_table('regression.txt', skipinitialspace=True)
df = df.iloc[1: , :]
df = df["x                    y"].str.split('             ',n=2,expand=True)
df = df.astype(float)
ds=df.to_numpy()

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)

#Function set (+-*/)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)

#Terminal set (X, random numbers between -1 and 1)
pset.renameArguments(ARG0='x')
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#initialization method is ramped half&half, tree depth is 2-6
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6) 
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#Define fitness function
def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(individual)
    # Evaluate the mean squared error between the expression and the real Y value
    sqerrors=[]
    for d in ds:
        sqerrors.append((func(d[0])-d[1])**2)
    #treat NaN values in sqerrors
    lensq=len(ds)
    for i in range(len(sqerrors)):
        if(math.isnan(sqerrors[i])):
            sqerrors[i]=0
            lensq=lensq-1  
    try:
        return math.fsum(sqerrors) / lensq,
    except ZeroDivisionError:
        return 100,  #If all values are nan type, this individial have a fitness of 100 which will be removed from the next generation
    

def testfunc(individual):
    func = toolbox.compile(individual)
    return func(ds[19][0])
    
toolbox.register("evaluate", evalSymbReg)
#select method - tournsize set to 7
toolbox.register("select", tools.selTournament, tournsize=7) #tournsize set to 7
#prepare to apply cross over
toolbox.register("mate", gp.cxOnePoint) 
#prepare to apply mutation - use full method to generate trees randomly for mutation, depth is 1-3
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3) 
#toolbox.register("expr_mut",gp.genHalfAndHalf,min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
#maximum tree depth set to 17
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) 
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) 

def main():
    random.seed(128) # Random seed for replicating the result - comment out or change the random seed to see different result.
    #set population size to 100
    pop = toolbox.population(n=100) 
    #select the best indivisual so far
    hof = tools.HallOfFame(5) 

    #set 80% cross over, 10% mutation, 51 generations
    CXPB, MUTPB, NGEN = 0.9, 0.1, 51
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    #Alternative way is to use the eaSimple algorithm - set 90% cross over, 5% mutation, 51 generations
    #NB: eaSimple algorithom doesn't do elitism!!!
    #pop, log = algorithms.eaSimple(pop, toolbox, 0.9, 0.05, 51, stats=mstats, halloffame=hof, verbose=True)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "bestfitness"
      
    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    
    #stats of the initial generation
    hof.update(pop)
    hof_size = len(hof.items) if hof.items else 0
    logbook.record(gen=0, evals=len(pop), bestfitness=hof[0].fitness)
    print(logbook.stream)

    for g in range(1, NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop) - hof_size) # tournzize=7
        # Clone the offspring
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
       
        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # add the best 5 individuals to the offspring
        elitism = tools.selBest(pop, 5) 
        offspring.extend(elitism)

        # Evaluate the individuals with an invalid fitness
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)
    
        # update and replacement of the population by the offspring
        hof.update(offspring)
        pop = offspring
        bestfit=hof[0].fitness.values
       
        logbook.record(gen=g, evals=len(invalids), bestfitness=bestfit)
        print(logbook.stream)

        if(bestfit[0]<0.1):
            break     
    print('Best individual : ', hof[0], 'Fitness:', bestfit[0])

    return pop, stats, hof
    

if __name__ == "__main__":
    main()
    


