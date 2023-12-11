import numpy as np
import pandas as pd

class StrategyOptimizerL:
    def __init__(self, fitness_function, n_generation, generation_size, n_gene, gene_range, mutation_probability, gene_mutation_probability, no_select_best):
        """
        Initialize the Strategy Optimizer.

        Parameters:
        fitness_function: Function to calculate the fitness of an individual.
        n_generation: Number of generations to run the algorithm.
        generation_size: Number of individuals in each generation.
        n_gene: Number of genes in each individual.
        gene_range: List of tuples specifying the min and max range of each gene.
        mutation_probability: Probability of mutating an individual.
        gene_mutation_probability: Probability of mutating each gene.
        no_select_best: Number of top individuals to select for the next generation.
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
        """ Generate a random individual. """
        individual = [np.random.randint(low, high) for low, high in self.gene_range]
        return individual

    def random_generate_population(self, population_size):
        """ Generate a random population. """
        return [self.random_generate_individual() for _ in range(population_size)]

    def random_mate_parent(self, parents, n_offspring):
        """ Generate offspring by mating parents. """
        n_parents = len(parents)
        offspring = []

        for _ in range(n_offspring):
            dad, mom = np.random.choice(parents, 2, replace=False)
            mask = np.random.randint(0, 2, size=len(dad))
            child = np.where(mask, dad, mom)
            offspring.append(child.tolist())

        return offspring

    def random_mutate_individual(self, individual):
        """ Mutate an individual. """
        for i in range(self.n_gene):
            if np.random.random() < self.gene_mutation_probability:
                gene_range = self.gene_range[i]
                individual[i] = np.random.randint(gene_range[0], gene_range[1])
        return individual

    def random_mutate_population(self, population):
        """ Mutate a population. """
        return [self.random_mutate_individual(ind) if np.random.random() < self.mutation_probability else ind for ind in population]

    def select_best_individuals(self, population, n_best):
        """ Select the best individuals from the population. """
        fitness_scores = [(self.fitness_function(ind), ind) for ind in population]
        sorted_population = sorted(fitness_scores, key=lambda x: x[0], reverse=True)
        best_individuals = [ind for _, ind in sorted_population[:n_best]]

        # Optional: Print fitness statistics
        best_score, avg_score, worst_score = sorted_population[0][0], np.mean([score for score, _ in fitness_scores]), sorted_population[-1][0]
        print(f'Best: {best_score}, Average: {avg_score}, Worst: {worst_score}')

        return best_individuals

    def run_algorithm(self):
        """ Run the genetic algorithm. """
        population = self.random_generate_population(self.generation_size)

        for _ in range(self.n_generation):
            best_individuals = self.select_best_individuals(population, self.no_select_best)
            offspring = self.random_mate_parent(best_individuals, self.generation_size - len(best_individuals))
            population = best_individuals + offspring
            population = self.random_mutate_population(population)

        return self.select_best_individuals(population, 1)[0]

# Example usage
# Define a fitness function
def sample_fitness_function(individual):
    # Example fitness function (to be customized)
    return sum(individual)

# Initialize and run the optimizer
optimizer = StrategyOptimizerL(
    fitness_function=sample_fitness_function,
    n_generation=10,
    generation_size=20,
    n_gene=5,
    gene_range=[(0, 100), (0, 100), (0, 100), (0, 100), (0, 100)],
    mutation_probability=0.1,
    gene_mutation_probability=0.05,
    no_select_best=5
)

best_individual = optimizer.run_algorithm()
print("Best Individual:", best_individual)
