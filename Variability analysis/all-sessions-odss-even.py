import sys
import os
# Get the current working directory
current_dir = os.getcwd()

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
    
from GridMetrics import GridScorer, circle_mask, get_even_odd_times, GridParameters, create_new_result_dir, load_grid_metrics_from_pickle
import json

rats = ['q', 'q', 'r1', 'r1', 'r1', 's', 'r2', 'r2', 'r2']
mods = ['1', '2', '1', '2', '3', '1', '1', '2', '3']


for rat, mod in zip(rats, mods):
    print('Computing metrics for rat ' + rat + ' mod ' + mod)
    G, general_results_working_directory, session_results_directory = load_grid_metrics_from_pickle(rat, mod)
    cell_trial_dict = G.compute_session_odd_even_metrics(n_trials=100, seconds_per_bin=60)

    filename = session_results_directory + '/' + rat + mod + '_odds-even.json'
    with open(filename, 'w') as f:
        json.dump(cell_trial_dict, f)