import random
import modules.constants as const


class Food:
	def __init__(self):
		self.position = (0,0)
		self.colour = (const.GREEN)
		self.randPosition()

	# the food spawns in random positions
	def randPosition(self):
		self.position = (random.randint(0, const.UNITS_WIDTH-1) * const.GRID, random.randint(0,const.UNITS_HEIGHT-1) * const.GRID)

	#checks if the food is eaten
	def isEaten(self, snake, food):
		if snake.actualHeadPosition() == self.position:
			snake.length += 1
			self.randPosition()
