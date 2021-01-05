from modules import food
from modules import snake
from modules import naiveAlgorithm as NA
import sys


if __name__ == '__main__':

	#initiate the objects
	snake = snake.Snake('Naive1.2-ng')
	food = food.Food() 

	#The number of trials is taken as command line arguments 
	max_trials = int(sys.argv[1])

	#Main Loop
	while snake.trials <= max_trials :
		
		#The function choose_move from the naiveAlgorithm script returns the calculated next move
		snake.chooseDirection(NA.choose_move(snake, food.position))
		snake.move()
		
		#for each iteration  we check if the snake and the food collided
		food.isEaten(snake,food)

	print('{} trials completed'.format(max_trials))
	sys.exit()
