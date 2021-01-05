import pygame
import modules.constants as const
from pygame.locals import *

screen = pygame.display.set_mode((const.WIDTH, const.HEIGHT), 0, 32) #creates a display surface with 32bits colours
surface = pygame.Surface(screen.get_size()) #surface on which the game is played
surface = surface.convert() #removes alpha pixels

def drawGrid(WIDTH, HEIGHT, GRID, surface):
	i = 0
	while i < WIDTH:
		pygame.draw.line(surface,(0,0,0),(i,0),(i,HEIGHT),1)
		pygame.draw.line(surface,(0,0,0),(0,i),(WIDTH,i),1)
		i += GRID

def drawSquare(surface, colour, pos):
	r = pygame.Rect((pos[0],pos[1]),(const.GRID, const.GRID))
	pygame.draw.rect(surface, colour, r) 

def drawSnake(surface, colour, positions):
	for p in positions:
		drawSquare(surface, colour, p)

def initiateGraphics():
	surface.fill(const.WHITE)
	drawGrid(const.WIDTH,const.HEIGHT, const.GRID,surface)

def updateGraphics(snake, food, fpsClock, fps):
	drawSnake(surface, const.COLOURS, snake.positions)
	drawSquare(surface, const.GREEN, food.position)
	screen.blit(surface,(0,0))
	pygame.display.flip()
	pygame.display.update()
	fpsClock.tick(fps) 