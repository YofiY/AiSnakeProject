from modules import constants as const
from modules import graphics
from modules import food
from modules import snake_object
from modules import naiveAlgorithm as NA
from modules import datalogger
import sys
import time
import random
import pygame
from pygame.locals  import *



#initiate pygame
pygame.init()

#sets up a pygame clock to limit the frame rate
fpsClock = pygame.time.Clock()  

#(delay, intervals [ms])
pygame.key.set_repeat(1,40) 

#frame rate limit
fps = const.FPS 

if __name__ == '__main__':

	#initiate the objects
	snake = snake_object.Snake('playable')
	food = food.Food()

	#main loop
	while True :
		#loop through events
		for event in pygame.event.get():
			
			if event.type == QUIT:
				datalogger.logAverageScore()
				pygame.quit()
				sys.exit()

			#seeks key inputs		
			elif event.type == KEYDOWN:

				# the 'n' and 'm' control the frame rate
				if event.key == K_n:
					fps -= 5
					
				if event.key == K_m:
					fps += 5
					
				#the snake's movement are dictated by the arrow keys
				if event.key == K_UP:
					snake.chooseDirection(const.UP)
				if event.key == K_DOWN:
					snake.chooseDirection(const.DOWN)
				if event.key == K_LEFT:
					snake.chooseDirection(const.LEFT)
				if event.key == K_RIGHT:
					snake.chooseDirection(const.RIGHT)
		#initiate graphics	
		graphics.initiateGraphics()

		#the snake takes the next the next position 
		snake.move()

		#check if the food is eaten	
		food.isEaten(snake,food)

		#the graphics are updated and the frame rate is controlled
		graphics.updateGraphics(snake, food, fpsClock, fps)

		#logs
		x, y = snake.positions[0]


	#log results
	datalogger.logAverageScore()
	pygame.quit()
	sys.exit()
