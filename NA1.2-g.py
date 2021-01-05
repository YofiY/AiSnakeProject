from modules import constants as const
from modules import graphics
from modules import food
from modules import snake_object
from modules import naiveAlgorithm as NA
from modules import datalogger
from modules import backtracking as BT
import sys
import time
import pygame
from pygame.locals  import *

#initiate pygame
pygame.init()

#sets up a pygame clock to limit the frame rate
fpsClock = pygame.time.Clock()

#(delay, intervals [ms])
pygame.key.set_repeat(1,40) 

last_moves = []

#frame rate limit
fps = const.FPS

if __name__ == '__main__':

	#initiate the objects
	snake = snake_object.Snake('Naive1.2-g')
	food = food.Food()
	
	#The number of trials is taken as a command line argument
	max_trials = int(sys.argv[1])

	#main loop
	while snake.trials <= max_trials:
		
		#loop through events
		for event in pygame.event.get():
			
			#quits the program if the user quits
			if event.type == QUIT:
				datalogger.logAverageScore('Naive1.2-g')
				pygame.quit()
				sys.exit()
			# the 'n' and 'm' control the frame rate
			elif event.type == KEYDOWN:

				if event.key == K_n:
					fps -= 5
					
				if event.key == K_m:
					fps += 5
		
		#initiate graphics			
		graphics.initiateGraphics()
		
		#The function choose_move() from the naiveAlgorithm script returns the calculated next move
		snake.chooseDirection(NA.choose_move(snake, food.position))
		
		#print(BT.find_path(snake, food.position, last_moves))
		#print(BT.is_food_around(snake, food.position))
		
		#the snake takes the next the next position 
		snake.move()
		
		#for each iteration  we check if the snake and the food collided
		food.isEaten(snake,food)

		#the graphics are updated and the frame rate is controlled
		graphics.updateGraphics(snake, food, fpsClock, fps)

	#log results into xml file
	datalogger.logAverageScore('Naive1.2-g')
	pygame.quit()
	sys.exit()
