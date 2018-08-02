import numpy as np
from itertools import islice
from tsp_utils import InstanceLoader, create_graph_metric, create_dataset_metric
from anneal import SimAnneal

if __name__ == '__main__':
  test_batches_per_epoch = 1
  batch_size = 1
  bins = 10 ** 6
  test_loader = InstanceLoader("test")
  with open('TSP-closest-anneal-log.dat','w') as logfile:
    print("inst_i\tinst_size\tclosest_fitness\tsa_fitness\tsa_iter", file=logfile)
    # Run for a number of epochs
    for epoch_i in range(1):

       
        sa_acc    = 0
        cn_acc    = 0

        print("Testing model...", flush=True)
        for (inst_i, inst) in enumerate(test_loader.get_instances(len(test_loader.filenames))):
            _, Mw, _ = inst
            Mw = np.round( Mw * bins )
            sa = SimAnneal( Mw )
            cn_acc += sa.best_fitness
            print( "{inst_i}\t{inst_size}\t{closest_fitness}\t".format(inst_i=inst_i, inst_size=Mw.shape[0], closest_fitness = sa.best_fitness), end="", file=logfile )
            sa.anneal()
            sa_acc += sa.best_fitness
            print( "{sa_fitness}\t{sa_iter}".format(sa_fitness=sa.best_fitness, sa_iter=sa.iteration), file=logfile )
            
        #end 
        cn_acc /= inst_i
        sa_acc /= inst_i 
        print("avg\ttotal\t{closest_avg}\t{sa_avg}\tNA".format(closest_avg=cn_acc, sa_avg=sa_acc), file = logfile )
    #end
