# Libraries
from reais_methods import run_sga
from reais_methods import run_cga
from reais_methods import sphere
from reais_methods import rosen
import matplotlib.pyplot as plt


# Mutual Parameters
maximum_generation = 100
population_size = 20
fitness_obj = sphere

# Reais sGA:
chromosome_length = 8
crossover_rate = 0.7
mutation_rate =  0.8
bounds = [(-100, 100)]

# To more high the dimension, more difficult to converge
for chromosome_length in [8, 16, 32, 64]:
    it = list(run_sga(bounds * chromosome_length,fitness_obj, its=3000))
    x, f = zip(*it)
    plt.plot(f, label='d={}'.format(chromosome_length))
plt.legend()
plt.show()

#-------------------------------------

# Reais cGA:
for chromosome_length in [8, 16, 32, 64]:
    best_score_progress_cga = run_cga(maximum_generation, chromosome_length,fitness_obj)
    plt.plot(best_score_progress_cga, label='d={}'.format(chromosome_length))
plt.legend()
plt.show()

