# This is necessary to find the main code
import sys, os
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

def main():
    # TODO This is your code!
    sys.path.insert(1, '../team02')
    from testcharacter_q_learning import TestCharacter

    # Create the game
    # random.seed(123) # TODO Change this if you want different random choices
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 5,     # position
    ))
    g.add_monster(SelfPreservingMonster("aggressive", # name
                                        "A",          # avatar
                                        3, 13,        # position
                                        2             # detection range
    ))

    # TODO Add your character
    g.add_character(TestCharacter("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))

    # Run!
    g.go(1)

if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)