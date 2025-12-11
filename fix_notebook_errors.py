import json
import os

def fix_notebook_file(nb_path):
    print(f"Reading {nb_path}...")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # Define calculate_chunked_stats function source code
    stats_func_code = [
        "\n",
        "def calculate_chunked_stats(history, chunk_size=50):\n",
        "    stats = []\n",
        "    n = len(history)\n",
        "    for i in range(0, n, chunk_size):\n",
        "        chunk = history[i:i+chunk_size]\n",
        "        success_count = sum(chunk)\n",
        "        failure_count = len(chunk) - success_count\n",
        "        success_rate = success_count / len(chunk) if chunk else 0\n",
        "        stats.append({\n",
        "            'chunk_idx': i // chunk_size,\n",
        "            'start_episode': i,\n",
        "            'end_episode': min(i + chunk_size, n) - 1,\n",
        "            'success_count': success_count,\n",
        "            'failure_count': failure_count,\n",
        "            'success_rate': success_rate\n",
        "        })\n",
        "    return stats\n"
    ]

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            full_source = "".join(source)
            
            # 1. Inject calculate_chunked_stats definition
            # We'll put it in the same cell as evaluate_agent for convenience
            if "def evaluate_agent" in full_source and "def calculate_chunked_stats" not in full_source:
                print("Found evaluate_agent calling cell. Injecting calculate_chunked_stats...")
                nb['cells'][i]['source'] = stats_func_code + source
                modified = True
            
            # 2. Replace plot_intermediate_results
            if "plot_intermediate_results" in full_source:
                print("Found plot_intermediate_results usage. Replacing...")
                new_source = []
                for line in source:
                    if "plot_intermediate_results" in line:
                        # Replace with plot_individual_performance
                        if "expert_history=successes" in line:
                             line = line.replace(
                                 "plot_intermediate_results(expert_history=successes, all_results=[], n_episodes=500)",
                                 "plot_individual_performance(\"Expert\", returns, successes)"
                             )
                        else:
                            line = line.replace("plot_intermediate_results", "plot_individual_performance")
                        new_source.append(line)
                    else:
                        new_source.append(line)
                nb['cells'][i]['source'] = new_source
                modified = True

    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook saved successfully.")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    notebooks = [
        "/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb",
        "/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb"
    ]
    for nb in notebooks:
        try:
            if os.path.exists(nb):
                fix_notebook_file(nb)
            else:
                print(f"Notebook not found: {nb}")
        except Exception as e:
            print(f"Error processing {nb}: {e}")
