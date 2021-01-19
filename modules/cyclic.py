import modules.constants as const



def change_direction(snake, food): #Returns a boolean, wether the condition necessary to change the direction are met
	rel_pos_snake_to_food = ((food.position[0] - snake.positions[0][0]) / 20, (food.position[1] - snake.positions[0][1]) / 20)
	"""
	if rel_pos_snake_to_food == (1,0):
		print('1 bloc away shortcut = {}'.format(rel_pos_snake_to_food))
		return True 
	"""
	if snake.positions[0][1] == 0 or snake.positions[0][1] / 20 == const.UNITS_HEIGHT - 1 :
		return True
	if snake.direction == const.LEFT or  snake.direction == const.RIGHT:
		return True
	else:
		return False

def new_direction(snake,food): #Once the necessary conditions are met, returns the next move following the cyclic path
	rel_pos_snake_to_food = ((food.position[0] - snake.positions[0][0]) / 20, (food.position[1] - snake.positions[0][1]) / 20)
	"""
	if rel_pos_snake_to_food == (1,0):
		return const.RIGHT
	"""
	if snake.positions[0][1] == 0 and snake.direction == const.UP:
		return const.RIGHT
	if snake.positions[0][1] == 0 and snake.direction == const.RIGHT:
		return const.DOWN
	if snake.positions[0][1] / 20 == const.UNITS_HEIGHT - 1 and snake.direction == const.DOWN:
		return const.RIGHT
	if snake.positions[0][1] / 20 == const.UNITS_HEIGHT - 1 and snake.direction == const.RIGHT:
		return const.UP
	if snake.direction == const.LEFT or snake.direction == const.RIGHT:
		return const.DOWN
































