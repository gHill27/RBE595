import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

monster_game = "test_monster.py"
total_runs = 100
# Run up to 10 games at the exact same time to speed things up
max_concurrent_workers = 1 

def run_game(script_name):
    exploration_prob = " 0.5"

    try:
        result = subprocess.run([sys.executable, script_name, exploration_prob], capture_output=True, text=True, timeout=10)
        output = result.stdout

    except subprocess.TimeoutExpired as e:
        output = e.stdout
        
    # returns whether the game ended in a win
    return 1 if "found the exit" in str(output) else 0

if __name__ == "__main__":
    total_wins = 0
    
    print(f"Running {monster_game} {total_runs} times...")
    
    # Open the thread pool ONCE to avoid performance bottleneck
    with ThreadPoolExecutor(max_workers=max_concurrent_workers) as executor:
        # Pass the script name 100 times into the executor
        results = executor.map(run_game, [monster_game] * total_runs)
        
        # Sum up all the 1s (wins) as the threads finish
        total_wins = sum(results)

    print(f"Total wins: {total_wins}/{total_runs}")