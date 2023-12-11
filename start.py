
import numpy as np
import pandas as pd


class StrategyOptimizerL:
    def __int__(self,
                fitness_function,
                n_generation,
                generation_size,
                n_gene,
                gene_range,
                mutation_probability,
                gene_mutation_probability,
                no_select_best
                ):
        """
        'fitness_function'
        'n_generation'
        'generation_size'
        'n_gene' = 5
        'gene_range' = [(3,100),(3,70),(),...,()]
        'mutation_probability'
        'gene_mutation_probability'
        'no_select_best'
        """
        self.fitness_function = fitness_function
        self.n_generation = n_generation
        self.generation_size = generation_size
        self.n_gene = n_gene
        self.gene_range = gene_range
        self.mutation_probability = mutation_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.no_select_best = no_select_best


    def random_generate_individual(self):
        """
        Crete individual
        """
        individual = []
        for i in range (self.n_gene):
            gene = np.random.randint(self.gene_range[i][0], self.gene_range[i][1])
            individual.append(gene)
        return individual

    def random_generate_population(self, population_size):
        """
        Crete population
        """
        popultaion = []
        for i in range(len(population_size)):
            popultaion.append(self.random_generate_individual())
        return popultaion
    def random_mate_parent(self, parents, n_offspring):
        """
        Crete population
        """

        n_parents = len(parents)
        offspring = []

        for i in range(n_offspring):
            random_dad = parents[np.random.randint(0, n_parents-1)]
            random_mom = parents[np.random.randint(0, n_parents - 1)]

            dad_mask = np.random.randint(0,2, size=np.array(random_dad).shape)
            mom_mask = np.logical_not(dad_mask)

            child = np.add(np.multiply(random_dad, dad_mask), np.multiply(random_mom, mom_mask))
            offspring.append(child)
        return offspring


    def random_mate_individual(self, individual):
        """
        Crete population
        """

        new_individual = []

        for i in range(0, self.n_gene):
            gene = individual[i]
            #mutate gene
            if np.random.random() < self.gene_mutation_probability:
                if np.random.random() < 0.5:
                    #brut force mutation
                    gene = np.random.randint(self.gene_range[i][0], self.gene_range[i][1])

                else:
                    #mutate nicer
                    left_range = self.gene_range[i][0]
                    right_range = self.gene_range[i][1]

                    gene_dist = right_range - left_range
                    x = individual[i] + gene_dist/3 * (2*np.random.random() -1 )

                    if x > right_range:
                        x = (x-left_range) % gene_dist+left_range
                    elif x < left_range:
                        x = (right_range-x) % gene_dist + left_range
                gene = int(x)
            new_individual.append(gene)
        return new_individual



    def random_mate_population(self,population):
        """
        Crete population
        """

        mutated_pop = []
        for individual in population:
            new_individual = individual
            if np.random.random() < self.mutation_probability:
                new_individual = self.random_mate_individual(individual)

            mutated_pop.append(new_individual)

        return mutated_pop

    def select_best_individuals(self, population, n_best):
        """

        """
        fitness= []
        for idx, individual in enumerate(population):
            individual_fitness = self.fitness_function(individual)
            fitness.append([idx, individual_fitness])
        cost_tmp  = pd.DataFrame(fitness).sort_values(by=1, ascending=False).reset_index(drop=True)
        selected_parents_idx = list(cost_tmp.iloc[:n_best,0])
        selected_parents = [parent for idx, parent in enumerate(population) if idx in selected_parents_idx]

        print(f'best is: {cost_tmp[1].max()}, average: {cost_tmp[1].mean()}, and worst: {cost_tmp[1].min()}')



    def run_algorithm(self):
        parent_gen = self.random_mate_population(self.generation_size)

        for i in range(self.n_generation):

            parent_gen = self.select_best_individuals(parent_gen, self.no_select_best)
            parent_gen = self.random_mate_parent(parent_gen, self.generation_size)
            parent_gen = self.random_mate_population(parent_gen)
        best_child = self.select_best_individuals(parent_gen,10)
        return best_child

