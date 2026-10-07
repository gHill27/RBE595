# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster

# TODO This is your code!
sys.path.insert(1, '../team02')
from testcharacter_q_learning import TestCharacter

# Create the game
# random.seed(123) # TODO Change this if you want different random choices
g = Game.fromfile('map_bomb.txt')
# g.add_monster(StupidMonster("stupid", # name
#                             "S",      # avatar
#                             3, 3      # position
# ))

# TODO Add your character
g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go()
