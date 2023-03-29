from time import time as current_time
from ising import experiment

experiment.run(1, 1)
start_time = current_time()
experiment.run(250000, 192)
print("Score: ", 100000/float(current_time() - start_time))