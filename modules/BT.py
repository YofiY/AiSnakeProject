import modules.constants as const
from modules.naiveAlgorithm import sort_list, definePossibleMoves, isDead




def is_dead(hidden_snake_positions):
    if hidden_snake_positions[0] in hidden_snake_positions[2:]:
        return True
    else: 
        return False

def is_blocked(hidden_snake):

    possible_moves = definePossibleMoves(const.MOVES_LIST, hidden_snake)
    
    for i in range(len(possible_moves)):
        if isDead(hidden_snake, possible_moves[i]):
            possible_moves.remove(possible_moves[i])
    if len(possible_moves) == 0:
        return True 
    return False

def is_one_move_away(hidden_snake_positions, food_position):
    global snake_to_food
    snake_to_food = ((food_position[0] - hidden_snake_positions[0][0]) / const.GRID,(food_position[1] - hidden_snake_positions[0][1]) / const.GRID)

    if snake_to_food in const.MOVES_LIST:
        return True
    else:
        return False

def try_path(hidden_snake_positions, food_position, last_decisions):
    pass

global final_path

def backtracking(hidden_snake, food_position, last_decisions, path):
    
    moves = definePossibleMoves(const.MOVES_LIST, hidden_snake)
    snake_to_food = ((food_position[0] - hidden_snake.positions[0][0]) / const.GRID,(food_position[1] - hidden_snake.positions[0][1]) / const.GRID)

    moves = sort_list(snake_to_food, moves)
    has_reached_food = 0
    
    if hidden_snake.positions[0] == food_position:
        return True, path
    
    if is_blocked(hidden_snake):
        print('BLoCKED')
        return -1, path[-1]
    
    else:
        while has_reached_food != -1:
            print(has_reached_food)
            path.append(moves[0])
            hidden_snake.chooseDirection(moves[0])
            hidden_snake.move()
            has_reached_food, path  = backtracking(hidden_snake, food_position, last_decisions, path)
            print(' TAMER')
            return False, path
            
            if has_reached_food == True:
                print('HAS REACHED FOOD final_path = {} path = {}'.format(final_path, path)) 
                break
                return True, path

            elif has_reached_food == -1:
                hidden_snake.mov_backwards(path[-1])
                path.pop()
                moves.pop(0)
                

            

         
        
