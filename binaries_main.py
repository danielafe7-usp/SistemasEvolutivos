# Libraries
from binaries_methods import run_sga
from binaries_methods import run_cga
from binaries_methods import one_max_fitness
from binaries_methods import ftrap_5
import matplotlib.pyplot as plt


# Mutual Parameters
chromosome_length = 10
population_size = 20
maximum_generation = 100
fitness_obj = one_max_fitness#ftrap_5

# Binarios sGA: One Max Problem (OMP).
crossover_rate = 0.7
mutation_rate = 0.009
#best_score_progress_sga = run_sga(chromosome_length,population_size,maximum_generation,crossover_rate,mutation_rate,fitness_obj)


#-------------------------------------


# Binarios cGA: One Max Problem (OMP).
#best_score_progress_cga = run_cga(maximum_generation, chromosome_length, population_size, fitness_obj)


# Plott : To more high the dimension, more difficult to converge
for chromosome_length in [5,10]:
    best_score_progress_sga = run_sga(chromosome_length,population_size,maximum_generation,crossover_rate,mutation_rate,fitness_obj)
    best_score_progress_cga = run_cga(maximum_generation, chromosome_length, population_size, fitness_obj)
    sga_line, = plt.plot(best_score_progress_sga, color="green", linewidth=1.5, linestyle="-", label='SGA')
    cga_line, = plt.plot(best_score_progress_cga, color="red", linewidth=1.5, linestyle="dashed", label='CGA')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

