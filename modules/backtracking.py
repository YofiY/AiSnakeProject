"""
import modules.constants as const



def changePositionInList(mylist, item, Newposition):
	mylist.remove(item)
	mylist.insert(Newposition, item)
	return mylist

def definePossibleMoves(moves):
	for k in moves:	
		if snake.isNextPositionFree(k):
			continue
		else:	
			moves.remove(k)

	return moves


def BetterOne(snake, food_position):

	moves = [const.UP,const.DOWN,const.RIGHT,const.LEFT]
	

	snake_position = snake.positions[0]
	snakeToFoodVector = (food_position[0]-snake_position[0], food_position[1] - snake_position[1])
	rel_x, rel_y = snakeToFoodVector
	

	if rel_x > 0:
		changePositionInList(moves,const.RIGHT, 0)
	elif rel_x < 0:
	 	changePositionInList(moves, const.LEFT, 0)
	if rel_y > 0:
		changePositionInList(moves, const.DOWN, 0)
	elif rel_y < 0:
		changePositionInList(moves, const.UP, 0) 

	for i in range(len(moves)):
		if NA.isDead(snake, moves[i]):
			continue
		else:
			rightMove = moves[i]
			break

	return rightMoves


if snake.position[0] != food_position:
	moves = BetterOne(snake,food_position)
	if isDead(snake, moves):
		while len(moves) < 2:
			positionHistory, snake.positions, moves = Backtrack(positionHistory, positions, moves)
		moves = moves[1:]






def set_path(snake, food_position):
	path = []
	moves = BetterOne(snake, food_position)

		if isDead(snake, defaultMove):
			AllpositionList = Update_AllpositionList(-1, AllpositionsList)
			snake.positions = AllpositionsList[-1]

		else:
			log_positions_list(positions, allPositions)
			path.append(moves[0])










def log_positions_list(positions, AllpositionsList):
	AllpositionList.append(positions)
	return AllpositionList


def Backtrack(AllpositionList):
	return AllpositionList[-1]

def Update_AllpositionList(currentIteration, AllpositionList):
	AllpositionList = AllpositionList[:currentIteration]
	return AllpositionList

def isDead(snake, move):
	x, y = move
	actualHeadPosition = snake.positions[0]
	nextHeadPosition = (((actualHeadPosition[0]+(x*const.GRID)) % const.WIDTH), (actualHeadPosition[1] + (y*const.GRID)) % const.HEIGHT )
	if len(snake.positions) > 2 and nextHeadPosition in snake.positions[2:]:
		return True
	else:
		return False
"""

import modules.constants as const
import random

def isDead(snake, move):
	x, y = move
	actualHeadPosition = snake.positions[0]
	nextHeadPosition = (((actualHeadPosition[0]+(x*const.GRID)) % const.WIDTH), (actualHeadPosition[1] + (y*const.GRID)) % const.HEIGHT )
	if len(snake.positions) > 2 and nextHeadPosition in snake.positions[2:]:
		return True
	else:
		return False






def changePositionInList(mylist, item, Newposition):
	mylist.remove(item)
	mylist.insert(Newposition, item)
	return mylist

def definePossibleMoves(moves, snake):
	for k in moves:
		if snake.length > 1 and (k[0]*-1, k[1]*-1) == snake.direction:
			moves.remove(k)
	return moves

def choose_move(snake, food_position):

	moves = [const.UP,const.DOWN,const.RIGHT,const.LEFT]
	

	snake_position = snake.positions[0]
	snakeToFoodVector = (food_position[0]-snake_position[0], food_position[1] - snake_position[1])
	rel_x, rel_y = snakeToFoodVector
	

	if rel_x > 0:
		moves =	changePositionInList(moves,const.RIGHT, 0)
	elif rel_x < 0:
	 	moves = changePositionInList(moves, const.LEFT, 0)
	if rel_y > 0:
		moves = changePositionInList(moves, const.DOWN, 0)
	elif rel_y < 0:
		moves = changePositionInList(moves, const.UP, 0) 

	moves = definePossibleMoves(moves, snake)
	rightMove = random.choice(moves)

	for i in range(len(moves)):
		if isDead(snake, moves[i]):
			continue
		else:
			rightMove = moves[i]
			break

	return moves



def is_food_around(snake, pos_food):
	x_pos_snake, y_pos_snake = snake.positions[0]
	x_pos_food, y_pos_food = pos_food
	
	if abs(x_pos_food - x_pos_snake) == 20: 
		if x_pos_food - x_pos_snake ==  -20:
			return True, (-1,0)
		else: 
			return True, (1,0)
	
	elif abs(y_pos_food - y_pos_snake) == 20:
		if y_pos_food - y_pos_snake == -20:
			return True, (0,-1)
		else:
			return True, (0,1)
	else:
		return False, (0,0)

def find_path(snake, pos_food, last_moves):
	moves_list = []
	move_to_make = 'none' 
	head_snake = snake.positions[0]
	food_around, move_to_make = is_food_around(snake, pos_food)
	possible_moves = choose_move(snake, pos_food)

	if (food_around):
		moves_list.append(move_to_make)
		return True, moves_list



	if len(possible_moves) == 0:
		return False, moves_list
	
	#TRY 1ST DECISION
	for i in range(len(possible_moves)):
		moves_list.append(possible_moves[i])
		last_moves.append(possible_moves[i])
		snake.chooseDirection(possible_moves[i])
		snake.move()

		has_reached_food, last_moves = find_path(snake, pos_food, last_moves)

		if (has_reached_food):
			return has_reached_food, last_moves
			break
		else: 
			last_moves.pop()
			snake.positions.pop()


















