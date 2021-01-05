import modules.constants as const
import random
import sys
import modules.datalogger as DT

class Snake:
	def __init__(self, usedAlgorithm):
		self.length = 1
		self.trials = 0								#trials counter set to 0
		self.lost()									
		self.colour = const.RED
		self.usedAlgorithm = usedAlgorithm
		self.direction = random.choice([const.UP,const.DOWN,const.LEFT,const.RIGHT])
		self.iteration = 0
		self.state = 'alive' #bool on wether the snake dies or not
		self.colour = random.choice(const.COLOURS)


	def actualHeadPosition(self):
		return self.positions[0]

	def lost(self):											
		#logs results into xml file for each try
		if self.length != 1:
			DT.logData(self.trials, self.length, const.UNITS, self.usedAlgorithm) 
			self.state = 'dead'
		else:
			self.state

		print('renait')
		
		#the snake's length is set up to 1 unit
		self.length = 1
		#increment the trials counter
		self.trials += 1	

		#the snake spawns in the middle 															
		self.positions = [((const.WIDTH/2),(const.HEIGHT/2))]

		# a random direction is given 
		self.direction = random.choice([const.UP,const.DOWN,const.LEFT,const.RIGHT])
		print(self.state)
	def chooseDirection(self, direction):

		#unables the snake to take the opposite direction to its current direction
		if self.length > 1 and (direction[0]*-1, direction[1]*-1) == self.direction:
			return
		else:
			self.direction = direction
			

	def move(self):

		#calculate the next position headposition
		actualHeadPosition = self.positions[0]
		x, y = self.direction
		nextHeadPosition = (((actualHeadPosition[0]+(x*const.GRID)) % const.WIDTH), (actualHeadPosition[1] + (y*const.GRID)) % const.HEIGHT )

		#check for collision 
		if len(self.positions) > 2 and nextHeadPosition in self.positions[2:]:
			self.state = 'dead'
			self.lost()

		else: #the new position is inserted into the positions list
			self.positions.insert(0, nextHeadPosition)
			#the tail's last position is removed from the list
			if len(self.positions) > self.length:
				self.positions.pop()

	def move_backwards(self, preceding_move):
		
		#calculate the next position headposition
		actual_tail = self.positions[-1]
		x, y = preceding_move
		next_tail = (((actual_tail[0] - (x*const.GRID)) % const.WIDTH), (actual_tail[1] - (y*const.GRID)) % const.HEIGHT )

		self.positions.append(nextHeadPosition)
		self.positions.pop(0)

			
	

	

		
