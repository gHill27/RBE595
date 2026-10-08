# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../team02')
from testcharacter_q_learning import TestCharacter

# Create the game
# random.seed(123) # TODO Change this if you want different random choices
g = Game.fromfile('map_Monster.txt')
g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 3,        # position
                                    1             # detection range
))


# TODO Add your character
g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              2, 0  # position
))

# Run!
g.go(1)
