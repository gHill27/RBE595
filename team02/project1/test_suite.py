import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

variants = ["variant1.py", "variant2.py", "variant3.py", "variant4.py", "variant5.py"]
variant_wins = [0, 0, 0, 0, 0]
num_batches_per_variant = [4, 4, 20, 20, 20]
num_thread_workers = 5

def run_variant(variant_name):
    try:
        result = subprocess.run([sys.executable, variant_name], capture_output=True, text=True, timeout=10)
        output = result.stdout

    except subprocess.TimeoutExpired as e:
        output = e.stdout
    # returns whether the game ended in a win
    return 1 if "found the exit" in str(output) else 0

# run num_batches*num_thread_workers tests for each variant
for idx, variant in enumerate(variants):
    for _ in range(num_batches_per_variant[idx]):
        with ThreadPoolExecutor(max_workers=num_thread_workers) as executor:
            results = executor.map(run_variant, [variants[idx]]*num_thread_workers)

        for result in results:
            # record win rate
            variant_wins[idx] += result
    print("variant " + str(idx+1) + " wins: " + str(variant_wins[idx])+"/"+str(num_batches_per_variant[idx]*num_thread_workers))