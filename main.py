from time import time as current_time
from ising import experiment

start_time = current_time()
res = experiment.run()
print("Score: ", 100000/float(current_time() - start_time))