import numpy as np
import sys
#from pygame.locals  import *
#import pygame
import matplotlib.pylab as plt 
from random import random 
from modules import constants as const
from modules import food
from modules import snake_object
from modules import naiveAlgorithm as NA
from modules import genetic_algorithm as GA
import math
#from modules import graphics

 
	
def calc_next_head_position(head_position, direction):
	x, y = direction 
	next_head_position = (((head_position[0]+(x*const.GRID)) % const.WIDTH), (head_position[1] + (y*const.GRID)) % const.HEIGHT )
	return next_head_position

def deltax_deltay(head_position, food_position):
	delta_x = (food_position[0] - head_position[0]) #/ const.WIDTH
	delta_y = (food_position[1] - head_position[1]) #/ const.HEIGHT

	return delta_x, delta_y

def generate_inputs(snake, food_position):    #generates input for absolute directions
	#MOVES_LIST = [UP, DOWN, LEFT, RIGHT]
	generated_input = []
	
	for a, b in const.MOVES_LIST:
		if calc_next_head_position(snake.positions[0], (a, b)) in snake.positions:
			generated_input.append(0)
		else:
			generated_input.append(1)
	
	delta_x, delta_y = deltax_deltay(snake.positions[0], food_position)
	
	generated_input.append(delta_y)
	generated_input.append(delta_x)
	#print('GENERATED INPUT = {}'.format(generated_input))

	return generated_input

def generate_inputs2(snake, food_position, relative_directions):    #generates input for relative directions
    generated_input = []
    #generated_input.append(snake.direction)    
    for a, b in relative_directions:
        if calc_next_head_position(snake.positions[0], (a, b)) in snake.positions:
            generated_input.append(0.)
        else:
            generated_input.append(1.)    

    delta_x, delta_y = deltax_deltay(snake.positions[0], food_position)    
    generated_input.append(delta_x)
    generated_input.append(delta_y)
    #print('GENERATED INPUT = {}'.format(generated_input))
    return generated_input


def get_output(inputs, neuralNet):
	output = neuralNet.forward_pass(inputs)
	#print('OUTPUT LAYER PRE-SOFTMAX = {}'.format(output))
	processed_output = neuralNet.process_output(output)
	return processed_output

def softmax(x):
		return (np.exp(x) / sum(np.exp(x)))


def train(nb_generations):

	nb_input_nodes = 6
	nb_layer1_nodes = 4
	nb_layer2_nodes = 4	
	nb_output_nodes = 4
	mutation_rate = 0.1

	population_size = 300
	performance = []
	
	for current_generation in range(nb_generations):
		
		if current_generation == 0:

			#print('generation {}'.format(current_generation))

			#Initialization
			population = GA.GeneticPopulation(population_size, nb_input_nodes, nb_layer1_nodes, nb_layer2_nodes, nb_output_nodes)
			population_genome = population.generate_population()
 
		 	#fitness assignement 
			score_list = fitness(population_genome, population_size, current_generation)
			print('best = {}  mean = {}  median {}'.format(np.amax(score_list), np.mean(score_list), np.median(score_list)))

			performance.append(score_list)

		 	#selection top 20%
			parent_indexes = selection(score_list, population_size)
			
			
			#print('best: {} {}'.format(score_list[parents_index[0]], score_list[parents_index[1]]))

			
		 	#crossover
			population = GA.ChildrenGeneration(population_genome, parent_indexes, score_list, population_size, mutation_rate, nb_layer1_nodes, nb_layer2_nodes, nb_output_nodes)
			population_genome = population.generate_population()

		else:
			print('generation {}'.format(current_generation))


		 	#fitness assignement 
			score_list = fitness(population_genome, population_size, current_generation)
			print('best = {}  mean = {} median {}'.format(np.amax(score_list), np.mean(score_list), np.median(score_list)))	
			performance.append(score_list)

		 	#selection top 20%
			parent_indexes = selection(score_list, population_size)
			
	
			#print('best: {} {}'.format(score_list[parents_index[0]], score_list[parents_index[1]]))

			
		 	#crossover
			population = GA.ChildrenGeneration(population_genome, parent_indexes, score_list, population_size, mutation_rate, nb_layer1_nodes, nb_layer2_nodes, nb_output_nodes)
			population_genome = population.generate_population()

	return performance


				

def fitness(population_genome, population_size, current_generation):
	
	score_list = np.array([])
	#fpsClock = pygame.time.Clock()
	fps = 6
	snake = snake_object.Snake('genetic_alg.xml')
	foodie = food.Food()

	for i in range(population_size):
		snake.nb_moves = 0
		while (snake.trials-1) == i and snake.nb_moves < 20000:
			inputs = generate_inputs(snake, foodie.position)
			output = get_output(inputs, population_genome[i])

			
			#graphics.initiateGraphics()	
			snake.move()
			foodie.isEaten(snake, foodie)
			snake.chooseDirection(output)
			#graphics.updateGraphics(snake, foodie, fpsClock, fps)

		score_list = np.append(score_list, snake.score)
	
	return score_list

def selection(score_list, population_size):
	top20percent = population_size // 5
	return np.argpartition(score_list, -top20percent)[-top20percent:]

if __name__ == "__main__":
	performance = train(100)
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
	plt.ylabel('score (snake length)')
	plt.show()
	print('\n ----------------- \n average score: \n first generation = {} \n last generation = {}'.format(average[1], average[-1]))
