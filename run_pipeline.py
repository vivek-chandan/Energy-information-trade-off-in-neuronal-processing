import subprocess
import time
import os

def run_script(script_name):
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True, 
            capture_output=False
        )
        duration = time.time() - start_time
        print(f" SUCCESS: {script_name} completed in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f" FAILURE: {script_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f" ERROR: Could not run {script_name}. Reason: {e}")
        return False

import sys

if __name__ == "__main__":

    os.makedirs('data/figures/validation', exist_ok=True)
    os.makedirs('data/figures/novel_analysis', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    scripts = [
        "experiments/01_validate_basic_hh.py",
        "experiments/02_validate_bursting.py",
        "experiments/03_frequency_sweep.py",
        "experiments/04_multi_scale_analysis.py",
        "experiments/05_burst_pattern_study.py",
        "experiments/06_energy_information.py",
        "experiments/07_parameter_sensitivity.py",
        "experiments/08_extended_analysis.py" 
    ]
    
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print("\nPipeline stopped due to error.")
            break
            
    print(f"\nPipeline Finished. {success_count}/{len(scripts)} scripts ran successfully.")
    print("Check 'data/figures/' for your results.")