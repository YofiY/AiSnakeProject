import modules.constants as const
import random


def changePositionInList(mylist, item, new_position):
	mylist.remove(item)
	mylist.insert(new_position, item)
	return mylist

def definePossibleMoves(moves, snake):
	for k in moves:
		if snake.length > 1 and (k[0]*-1, k[1]*-1) == snake.direction:
			moves.remove(k)
	return moves

def sort_list(snake_to_food, moves):
	rel_x, rel_y = snake_to_food
	if rel_x > 0:
		if const.RIGHT in moves:
			moves =	changePositionInList(moves,const.RIGHT, 0)
	elif rel_x < 0:
		if const.LEFT in moves:
	 		moves = changePositionInList(moves, const.LEFT, 0)
	if rel_y > 0:
		if const.DOWN in moves:
			moves = changePositionInList(moves, const.DOWN, 0)
	elif rel_y < 0:
		if const.UP in moves:
			moves = changePositionInList(moves, const.UP, 0) 
	return moves

def snake_to_food_vector(snake_positions, food_position):
	return(food_position[0]-snake_positions[0][0], food_position[1] - snake_positions[0][1])
			
def choose_move(snake, food_position):

	moves = [const.UP,const.DOWN,const.RIGHT,const.LEFT]
	snake_to_food = snake_to_food_vector(snake.positions, food_position)
	moves = sort_list(snake_to_food, moves)
	moves = definePossibleMoves(moves, snake)
	rightMove = random.choice(moves)

	for i in range(len(moves)):
		if isDead(snake, moves[i]):
			continue
		else:
			rightMove = moves[i]
			break

	return rightMove

def isDead(snake, move):
	x, y = move
	actualHeadPosition = snake.positions[0]
	nextHeadPosition = (((actualHeadPosition[0]+(x*const.GRID)) % const.WIDTH), (actualHeadPosition[1] + (y*const.GRID)) % const.HEIGHT )
	if len(snake.positions) > 2 and nextHeadPosition in snake.positions[2:]:
		return True
	else:
		return False








