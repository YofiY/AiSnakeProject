from xml.dom import minidom
import xml.etree.ElementTree as ET
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

"""
The data's structure
------------------

		Try(nb)/
			Snakelength
			gridDimension
			Score = Snake length/gridDimension

------------------

<tryN>
	</snakeLength length = x>
	</gridDimension dimension = y>
	</score score = x/y>

</tryN>

"""
	
def logData(try_nb, snakeLength, gridDimension, usedAlgorithm):

	#the gameLog element is the root element
	gameLog = ET.Element('gameLog')
	str_gameLog = ET.tostring(gameLog, encoding = 'unicode', method = 'xml')
	
	Try = ET.SubElement(gameLog, 'Try')
	Try.set('nb', str(try_nb))

	#The snakeLength elemet takes the length value of the snake before it died
	element1 = ET.SubElement(Try, 'snakeLength')
	element1.set('length', str(snakeLength))

	#The gridDimension element takes the dimension values in units to establish the score
	element2 = ET.SubElement(Try, 'gridDimension')
	element2.set('dimension', str(gridDimension))

	#The score element is the proportion of the disponible units that the snake occupied before dying. The closer it gets to 1, the better.
	element3 = ET.SubElement(Try, 'Score')
	element3.set('score', str(snakeLength/gridDimension))

	#file openend for the first time in writing mode, so that if the file isn't created yet, one is created.
	#if there is data from previous sessions in the file, it is erased
	#the filename is defined by the algorithm used in the form 'data_AlgorithmName'
	#on the first opening the root element is written
	if try_nb == 1 :
		file = open('Data\data_{}.xml'.format(usedAlgorithm), 'w')
		file.write(str_gameLog+'\n')

	#the rest of the time the file is opened in append mode, so that i
	else:
		file = open('Data\data_{}.xml'.format(usedAlgorithm), 'a')
	
	#the data is converted to a string from the root element
	Try = ET.tostring(Try, encoding = 'unicode', method = 'xml')
	
	#the data is written in the file
	file.write(str(Try)+'\n')

	#the file is closed
	file.close()

def logAverageScore(usedAlgorithm):
	tree = ET.parse('Data\data_{}.xml'.format(usedAlgorithm))
	for node in tree.getroot():
		print(node)
	print(scoreList)











