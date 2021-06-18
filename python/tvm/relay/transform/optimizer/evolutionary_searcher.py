#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools

from .custom_fusion_pass import measure_end_to_end_user_defined
from ..backend_operator.utils import *
from workloads.relay_workloads import get_network_from_relay
from .op_match_logger import OpMatchLogger
from ..backend_operator.target import BEST_MATCH_LOG

# the goal ('fitness') function to be maximized
class EvolutionarySearcher:
    def __init__(self, op_state_to_match_translator, expr, net_name,
                 n_ops, pop_size=100, cx_prob=0.5, mut_prob=0.2, max_iter=100):

        self.op_state_to_match_translator = op_state_to_match_translator
        self.op_match_logger = OpMatchLogger()

        # duplicate checker
        self._memo_state = {}
        
        # Load network to measure
        self.net_name = net_name
        self.mod, self.params = get_network_from_relay(net_name, 1)
        self.expr = expr

        self.target_str = 'cuda'
        self.shape_dict = {"data": [1, 64, 56, 56]}

        # Hyperparameters
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.max_iter = max_iter

        # Prepare creators
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Attribute generator
        #                      define 'gen_state' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        self.toolbox = base.Toolbox()
        self.toolbox.register("gen_state", random.randint, 0, 1)

        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 100 'gen_state' elements ('genes')
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self.toolbox.gen_state, n_ops)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.measure_comp_graph)

        # register the crossover operator
        self.toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # self.toolbox.register("select", tools.selBest)

    def get_hash_of_individual(self, individual):
        return "".join((map(str, individual)))

    def update_best_individual(self, best_ind, cur_pop_best_ind):
        if best_ind == None or best_ind[0] < cur_pop_best_ind[0]:
            best_ind = cur_pop_best_ind

        return best_ind

    def measure_comp_graph(self, individual):
        print("measure_comp_graph" + "-"*30)
        # Note that the type of individual is (defined as) list

        # If this individual was measured before, we can skip
        # Warning(@Soo): If it takes up too much memory, we can comment this out
        individual_hash = self.get_hash_of_individual(individual)
        if individual_hash in self._memo_state:
            print(f"[Evaluation] Individual({individual}) was measured before ")
            return self._memo_state[individual_hash],

        # Translate individual into match
        print(f"[Evaluation] Individual: {individual}")
        opt_match = self.op_state_to_match_translator.translate(individual)
        # print(f"opt_match: {opt_match}")

        # Dump this opt_match in to files so that build pipeline can read it
        # USER_DEFINED_MATCH_LOG
        self.op_match_logger.save(self.expr, opt_match)
        print(f"[Evaluation] Match log saved")
        # Measure entire computation graph with opt_match
        mean_perf, std_perf = measure_end_to_end_user_defined(self.mod["main"], self.params,
                                                              self.target_str, self.shape_dict)
        print(f"[Evaluation] individual {individual} perf: {mean_perf} ")

        self._memo_state[individual_hash] = -mean_perf

        return -mean_perf,
        # return sum(individual),

    def search(self, rnd_seed = 64):
        # Initialize
        random.seed(rnd_seed)

        # Pair of individual and fitness score (negative inference time)
        best_ind = None
        best_opt_match = None

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=self.pop_size)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        # CXPB, MUTPB = 0.5, 0.2

        print("Starting evolutionary search")

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Begin the evolution
        while g < self.max_iter:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("Current generation statistics")
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            # Best will choose individual with the biggest negative inference time
            cur_pop_best_ind = tools.selBest(pop, 1)[0]
            # cur_pop_best_ind = tools.selWorst(pop, 1)[0]
            cur_pop_best_ind = (cur_pop_best_ind, cur_pop_best_ind.fitness.values)
            best_ind = self.update_best_individual(best_ind, cur_pop_best_ind)

            # Dump the best match
            best_opt_match = self.op_state_to_match_translator.translate(best_ind[0])
            best_match_log_path = f"{BEST_MATCH_LOG}_{self.net_name}.log"
            self.op_match_logger.save(self.expr, best_opt_match, log_path = best_match_log_path)
            print(f"Best individual is {best_ind}")

        print("-- End of (successful) evolution --")

        print(f"Final best individual is {best_ind}")
        # print(self.op_state_to_match_translator.optimized_match)

        print("-"*30)
        print(best_opt_match)
        return best_opt_match