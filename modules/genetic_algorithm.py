import numpy as np
import matplotlib.pyplot as plt
import random

class NodeLayer:
	def __init__(self, nb_nodes, nb_inputs):
		self.nb_nodes = nb_nodes
		self.nb_inputs = nb_inputs
		self.weights = np.random.uniform(low = -1, high = 1, size=(nb_inputs, nb_nodes))
		self.bias = np.array([0.01])
		#self.bias = np.random.normal(-0.5, 0.5, size=(1, nb_nodes))
		#print('\n {} \n'.format(self.weights))

class NeuralNetwork:
	def __init__(self, layer1, layer2, layer3):
		self.layer1 = layer1
		self.layer2 = layer2
		self.layer3 = layer3
	
	def sigmoid(self, x):
		return 1 / (1+np.exp(-x))
	
	def tanh(self, x):
		return np.tanh(x)

	def softplus(self, x):
		return np.log(1+np.exp(x))

	def softmax(self,x):
		return (np.exp(x) / sum(np.exp(x)))

	def hard_elish(self, x):
		if x >= 0:
			return 1 / (1+np.exp(-x))
		else:
			return (np.exp(x)-1) / (1+np.exp(-x))

	def forward_pass(self, inputs):
		#Sum of dot product of each layer
		#Apply activation function to the layer sums
		
		z_layer_1 = np.dot(inputs, self.layer1.weights) 
		output_layer_1 = np.array(self.tanh(z_layer_1) + np.array([1]))

		z_layer_2 = np.dot(output_layer_1, self.layer2.weights) 
		output_layer_2 = np.array(self.tanh(z_layer_2) + np.array([1]))

		z_layer_3 = np.dot(output_layer_2, self.layer3.weights) 
		output_layer_3 = np.array(self.tanh(z_layer_3)) 
		
		#print('ouput_layer = {} z_layer3 = {}'.format(output_layer_3, z_layer_3))
		
		return z_layer_3

	def process_output(self, output):
		possible_decisions = [(0,-1), (0,1), (1,0), (-1,0)] # [NORTH, SOUTH, EAST, WEST]
		softmaxed_output = self.softmax(output.flatten())
		softmaxed_output = softmaxed_output.tolist()
		decision = np.random.choice([0, 1, 2, 3], p=softmaxed_output)
		return possible_decisions[decision]
		#return possible_decisions[np.argmax(output)]

class GeneticPopulation:
	def __init__(self, population_size, nb_input_nodes, nb_layer1_nodes, nb_layer2_nodes, nb_output_nodes):
		self.population_size = population_size #nb of individual per generation
		self.nb_input_nodes  = nb_input_nodes
		self.nb_layer1_nodes = nb_layer1_nodes
		self.nb_layer2_nodes = nb_layer2_nodes
		self.nb_output_nodes = nb_output_nodes
		#self.genome = generate_population
	
	def generate_population(self):
		population_genome = [] #referencing all brains of the generation
		
		for i in range(self.population_size): 
			#random generation of brains consited of 1 input, 2 hidden and 1 output layer
			layer1 = NodeLayer(self.nb_layer1_nodes, self.nb_input_nodes)
			layer2 = NodeLayer(self.nb_layer2_nodes, self.nb_layer1_nodes)
			layer3 = NodeLayer(self.nb_output_nodes, self.nb_layer2_nodes)
			
			population_genome.append(NeuralNetwork(layer1, layer2, layer3))
		self.genome = population_genome
		return population_genome


class ChildrenNodeLayer:
	def __init__(self, weights, nb_nodes):
		self.weights = weights
		self.bias = self.bias = np.array([0.01])



class ChildrenGeneration:
	def __init__(self, parent_population_genome, parent_indexes, parent_score_list, population_size, mutation_rate, nb_layer1_nodes, nb_layer2_nodes, nb_output_nodes):
		self.parent_population_genome = parent_population_genome
		self.parent_indexes = parent_indexes
		self.parent_score_list = parent_score_list
		self.population_size = population_size
		self.mutation_rate = mutation_rate
		self.nb_layer1_nodes = nb_layer1_nodes
		self.nb_layer2_nodes = nb_layer2_nodes
		self.nb_output_nodes = nb_output_nodes
		
	def crossover(self, dad_layer, mother_layer, alpha): #takes mum and dad layer as input and returns list of child layer
		beta  = 1 - alpha
		weighted_mother_layer = np.dot(mother_layer, beta)
		weighted_father_layer = np.dot(dad_layer, alpha)
		kid = weighted_father_layer.__add__(weighted_mother_layer)

		return kid


	def selection(self): #Two-point roulette wheel selection
		is_parent = np.zeros(self.population_size).astype(int) 
		np.put(is_parent, self.parent_indexes, 1) #Is parent is an array of len=popsize of booleans on wether the individual at the given index is selected as a parent


		S = np.sum(self.parent_score_list, where=is_parent.astype(bool))
		P = random.randint(1, S)
		partial_sum = 0
		i = 0
		while partial_sum < 2*P:
			partial_sum += (self.parent_score_list[self.parent_indexes[i]] + self.parent_score_list[self.parent_indexes[-i]])
			i = (i+1)%len(self.parent_indexes)
		
		return (self.parent_indexes[i], self.parent_indexes[-i])
	"""			
	def mutation(self, kid, mutation_rate): #whole mutation 
		for i in range(len(kid)):
			if random.random() < mutation_rate:
				kid[i] = random.uniform(-1,1)
		return kid
	"""		
	def mutation(self, layer): #whole mutation
		
		for array in range(len(layer)):
			for element in range(len(layer[array])):
				if random.random() < self.mutation_rate:
					layer[array][element] = random.uniform(-1,1)
		return layer			
		
	def generate_population(self): #20% of the fittest from past generation, 20% mutated ,80% offspring
		population_genome = []
	
		#We include the selected parents in the next generation (20%of popsize)
		for selected_parent in self.parent_indexes:
			population_genome.append(self.parent_population_genome[selected_parent]) #20% of next generation

			layer1_weights = self.mutation(self.parent_population_genome[selected_parent].layer1.weights)
			layer2_weights = self.mutation(self.parent_population_genome[selected_parent].layer2.weights)
			layer3_weights = self.mutation(self.parent_population_genome[selected_parent].layer3.weights)

			layer1 = ChildrenNodeLayer(layer1_weights, self.nb_layer1_nodes)
			layer2 = ChildrenNodeLayer(layer2_weights, self.nb_layer2_nodes)
			layer3 = ChildrenNodeLayer(layer3_weights, self.nb_output_nodes)

			population_genome.append(NeuralNetwork(layer1, layer2, layer3))
			 #20% of next generation MUTATED PARENTS

		previous = []
		#We add the offsprings (80% of popsize)
		for i in range(self.population_size - (self.population_size // 5)):
			parents = self.selection()
			if parents[0] == parents[1]:
				parents = self.selection()

			if parents in previous:
				parents = self.selection()
			previous.append(parents)
			mother = self.parent_population_genome[parents[0]]
			father = self.parent_population_genome[parents[1]]
		
			alpha = random.random()

			layer1_weights = self.crossover(mother.layer1.weights, father.layer1.weights, alpha)
			layer2_weights = self.crossover(mother.layer2.weights, father.layer2.weights, alpha)
			layer3_weights = self.crossover(mother.layer3.weights, father.layer3.weights, alpha)

			layer1 = ChildrenNodeLayer(layer1_weights, self.nb_layer1_nodes)
			layer2 = ChildrenNodeLayer(layer2_weights, self.nb_layer2_nodes)
			layer3 = ChildrenNodeLayer(layer3_weights, self.nb_output_nodes)

			population_genome.append(NeuralNetwork(layer1, layer2, layer3))

		return population_genome
				

def selection(score_list, population_size):
	top20percent = population_size // 5
	return np.argpartition(score_list, -top20percent)[-top20percent:]

if __name__ == "__main__":
	performance = train(600)
	average = []
	median = []
	best_scores = []

	for gen in range(len(performance)):
	    average.append(np.average(performance[gen]))
	    median.append(np.median(performance[gen]))
	    best_scores.append(np.amax(performance[gen]))

	plt.plot(median, label = 'median')	
	plt.plot(average, label = 'average')
	plt.plot(best_scores, label = 'best_scores')
	plt.legend(loc = 'upper right')
	plt.xlabel('generation')
	plt.show()
	print('\n ----------------- \n average score: \n first generation = {} \n last generation = {}'.format(average[1], average[-1]))
			
