import argparse
import os

space = '\n'*100
logo = r""" 

  _______   _                           _    _____                   _           
 |__   __| | |                  /\     (_)  / ____|                 | |          
    | |    | |__     ___       /  \     _  | (___    _ __     __ _  | | __   ___ 
    | |    | '_ \   / _ \     / /\ \   | |  \___ \  | '_ \   / _` | | |/ /  / _ \
    | |    | | | | |  __/    / ____ \  | |  ____) | | | | | | (_| | |   <  |  __/
    |_|    |_| |_|  \___|   /_/    \_\ |_| |_____/  |_| |_|  \__,_| |_|\_\  \___|
                                                                                 
                                                                                 

Welcome, you are now in the Ai Snake's World. For further information enter: "menu.py --help"
To access the source code, visit github.com/YofiY/AiSnakeTM

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------

"""

print(space, logo)

#Lists all disponible AI models to solve the game
aimodels = ['NA1.2']

# Initiates the parser
parser = argparse.ArgumentParser(prog='AiSnake')

#the -g & -G arguments are put in mutually exclusive groups so that it is not possible to call them at the same time
group1 = parser.add_mutually_exclusive_group()
group1.add_argument('-g','--nongraphical', help='unables the graphical interface.', action='store_true')
group1.add_argument('-G','--graphical', help='enables the graphical interface', action='store_true')
parser.add_argument('-P', '--playable', help='start a playable version of the game', action='store_true')
parser.add_argument('-AI', '--model', type=str, help='start an AI played version of the game. Must be followed by the ai\'s model\'s name', action='store', default='Naive1.2')
max_trials =parser.add_argument('-T', '--trials', type=int, help='specify the N number of trials over which the snake is allowed to play', action='store',default=5)

#parses the arguments into the args variable
args = parser.parse_args()

if args.playable:
	print('lunching a playable version of the game...')
	os.system('playable_main.py')

elif args.graphical: 
	if args.model in aimodels:
		print('luching a graphical interface of the game with the model: {} given {} trials'.format(args.model, args.trials))
		#the -g in the files name implies that it is a graphical version
		os.system('{}-g.py {}'.format(args.model, args.trials))
	else:
		print('this ai model name is not attributed')

elif args.nongraphical:
	if args.model in aimodels:
		print('luching a non-graphical interface of the game with the model: {} given {} trials'.format(args.model, args.trials))
		#the -ng in the file's name implies that it is a nongraphical version 
		os.system('{}-ng.py {}'.format(args.model, args.trials))
	else:
		print('this ai model name is not attributed')


	

def set_max():
	return args.trial