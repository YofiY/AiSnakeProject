import os
import neat
import pygame

from modules import graphics
from modules import snake_object as snk 
from modules import food as fd
from modules import constants as const 

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    
    winner = p.run(eval_genome, 100)
    
    print('\nBest genome:\n{!s}'.format(winner))

def eval_genome(genomes, config):
    nets = []
    snakes = []
    ge = []
    food = fd.Food()
    
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snakes.append(snk.Snake('NEAT'))
        ge.append(genome)

    pygame.init()
    fpsClock = pygame.time.Clock()  
    for i in range(len(snakes)):
        while snakes[i].state == 'alive' and snakes[i].nb_moves < 200:
            

            ge[i].fitness = snakes[i].score
            relative_directions = const.RELATIVE_POSITIONS_DICTIONARY[str(snakes[i].direction)]
            #list relative to the current direction ['FORWARD', 'LEFT', 'RIGHT']

            inputs = generate_inputs(snakes[i], food.position, relative_directions)
            output = nets[i].activate(inputs) # output in the form [FORWARD, LEFT, RIGHT]

            decision_index = output.index(max(output))
            move = relative_directions[decision_index]

            graphics.initiateGraphics()
                
            snakes[i].move()

            food.isEaten(snakes[i], food)

            snakes[i].chooseDirection(move)

            graphics.updateGraphics(snakes[i], food, fpsClock, const.FPS)

def calc_next_head_position(head_position, direction):
    x, y = direction 
    next_head_position = (((head_position[0]+(x*const.GRID)) % const.WIDTH), (head_position[1] + (y*const.GRID)) % const.HEIGHT )
    return next_head_position

def deltax_deltay(head_position, food_position):
    delta_x = (food_position[0] - head_position[0]) / const.WIDTH
    delta_y = (food_position[1] - head_position[1]) / const.HEIGHT
    return delta_x, delta_y

def generate_inputs(snake, food_position, relative_directions):    
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

   

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'modules/config-feedforward.txt')
    run(config_path)
