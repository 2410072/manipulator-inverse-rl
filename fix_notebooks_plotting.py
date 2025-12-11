import json
import os

# Function to update a single notebook
def update_notebook(notebook_path, algo_func_name, plot_func_name, target_plot_header):
    if not os.path.exists(notebook_path):
        print(f"Skipping {notebook_path}, file not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_algo = 0
    changes_plot = 0

    # --- Part 1: Reorder Algorithm Loop ---
    # Strategy: Find the loop cell and move evaluate_agent BEFORE results collection
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = "".join(source)
            
            # Identify the Training Loop Cell (approx heuristic based on variable names usually present)
            # IRL: "def projection_method_algorithm" - handled inside function
            # GAIL: "for run_idx in range(max_runs):" - handled in global scope in cells
            
            is_algo_cell = False
            if "def projection_method_algorithm" in source_str and algo_func_name == "projection_method_algorithm": # IRL
                is_algo_cell = True
            elif "for run_idx in range(max_runs):" in source_str and algo_func_name == "gail_loop": # GAIL
                is_algo_cell = True
            
            if is_algo_cell:
                new_source = []
                eval_idx = -1
                live_plot_idx = -1
                
                # Scan for markers
                # Common marker in both: "Live Plotting" or similar comment
                # GAIL might use "all_results.append"
                
                # Markers
                eval_marker = "evaluate_agent"
                collection_marker = "all_results.append"
                
                # Find current positions
                for i, line in enumerate(source):
                    if eval_marker in line and "def " not in line: # Exclude definition
                        eval_idx = i
                    if collection_marker in line:
                        live_plot_idx = i
                        
                # Determine Eval Block (usually 2-4 lines)
                if eval_idx != -1:
                    eval_start = eval_idx
                    # Heuristic: Eval block often starts with a print statement before it
                    if eval_start > 0 and "print" in source[eval_start-1] and "Evaluation" in source[eval_start-1]:
                        eval_start -= 1
                    
                    eval_end = eval_idx + 1 # Include the print after?
                    if eval_end < len(source) and "print" in source[eval_end] and "Success Rate" in source[eval_end]:
                        eval_end += 1
                        
                    eval_lines = source[eval_start:eval_end+1]
                    
                    # If Eval is AFTER Collection, we move it
                    # Logic assumes we want eval before collection to store it
                    
                    # Construction for GAIL (loop based) and IRL (func based) slightly differs but logic is same:
                    # 1. Pre-collection code
                    # 2. Eval Code
                    # 3. Collection Code (with injected 'eval_sr')
                    # 4. Post-processing
                    
                    # Simplified reconstruction:
                    # We will delete the original eval lines and insert them before 'res = {' or 'all_results.append'
                    
                    # Find insertion point (start of result dictionary construction)
                    insert_idx = -1
                    for i, line in enumerate(source):
                        if "res = {" in line or "all_results.append" in line: # IRL / GAIL
                            insert_idx = i
                            # Backtrack to catch "final_sr = " if part of collection block
                            if i > 0 and "final_sr =" in source[i-1]:
                                insert_idx = i-1
                            break
                    
                    if insert_idx != -1 and insert_idx < eval_start:
                        # Construct new list
                        # 1. Everything before insertion point
                        new_source = source[:insert_idx]
                        
                        # 2. Eval Lines
                        new_source.extend(eval_lines)
                        new_source.append("\n")
                        
                        # 3. Middle chunk (Insertion Point -> Eval Start)
                        middle_chunk = source[insert_idx:eval_start]
                        
                        # Modify middle chunk to include stats if dictionary definition is there
                        for line in middle_chunk:
                            if "\"chunk_stats\":" in line or "\"success_history\":" in line: # Inside dict
                                new_source.append(line)
                                if  "\"chunk_stats\":" in line: # Good anchor
                                    new_source.append("            \"noise_free_success_rate\": eval_sr,\n")
                            elif "res = {" in line:
                                new_source.append(line)
                            elif "all_results.append" in line and "res" not in line: # GAIL might append simple dict
                                # GAIL specific handling if it builds dict inline
                                # Assuming GAIL builds 'res' before appending like IRL for consistency?
                                # Looking at previous file view for GAIL... 
                                # It seems GAIL cell wasn't fully shown but typically follows pattern.
                                # Safe bet: If using 'res =', inject there.
                                new_source.append(line)
                            else:
                                new_source.append(line)
                                
                        # 4. End chunk (After Eval Block)
                        if eval_end + 1 < len(source):
                            new_source.extend(source[eval_end+1:])
                            
                        cell['source'] = new_source
                        changes_algo = 1
                        print(f"[{algo_func_name}] Reordered evaluation logic.")

    # --- Part 2: Modify Plotting Function ---
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = "".join(source)
            
            if plot_func_name in source_str:
                new_source = []
                for line in source:
                    # 1. Layout
                    if "nrows=2" in line:
                        line = line.replace("nrows=2", "nrows=3").replace("figsize=(12, 12)", "figsize=(12, 18)")
                        new_source.append(line)
                    
                    # 2. Insert Plot 3
                    elif target_plot_header in line or "plt.tight_layout" in line:
                        if target_plot_header in line: # Already exists?
                             new_source.append(line)
                             continue
                             
                        # Insert before tight_layout
                        code_to_add = [
                            "\n",
                            "    # --- 3. Noise-Free Evaluation Comparison ---\n",
                            "    ax3 = axes[2]\n",
                            "    \n",
                            "    # Expert Benchmark (Static)\n",
                            "    if expert_history:\n",
                            "        exp_sr = np.mean(expert_history) * 100\n",
                            "        ax3.axhline(y=exp_sr, color='black', linestyle='--', linewidth=2, label=f'Expert Benchmark ({exp_sr:.1f}%)')\n",
                            "    \n",
                            "    # Apprentices (Noise-Free)\n",
                            "    app_ids = []\n",
                            "    app_scores = []\n",
                            "    for res in all_results:\n",
                            "        if 'noise_free_success_rate' in res:\n",
                            "            app_ids.append(res['id'])\n",
                            "            app_scores.append(res['noise_free_success_rate'])\n",
                            "    \n",
                            "    if app_ids:\n",
                            "        ax3.plot(app_ids, app_scores, marker='o', markersize=8, linestyle='-', linewidth=2, color='blue', label='Apprentice (Noise-Free)')\n",
                            "        for x, y in zip(app_ids, app_scores):\n",
                            "            ax3.annotate(f'{y:.1f}%', (x, y), textcoords=\"offset points\", xytext=(0,10), ha='center')\n",
                            "    \n",
                            "    ax3.set_title(\"Noise-Free Evaluation: True Model Capability vs Expert\")\n",
                            "    ax3.set_xlabel(\"Apprentice Generation\")\n",
                            "    ax3.set_ylabel(\"Success Rate (%)\")\n",
                            "    ax3.set_ylim(0, 105)\n",
                            "    ax3.grid(True, alpha=0.3)\n",
                            "    ax3.legend(loc='lower right')\n",
                            "\n"
                        ]
                        new_source.extend(code_to_add)
                        new_source.append(line)
                    elif "rect=[0, 0, 0.85, 1]" in line:
                        new_source.append(line)
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                changes_plot = 1
                print(f"[{algo_func_name}] Modified plotting function.")

    if changes_algo or changes_plot:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {notebook_path}")
    else:
        print(f"No suitable patterns found in {notebook_path}")

# Run for IRL
update_notebook(
    '/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb',
    "projection_method_algorithm",
    "def plot_intermediate_results",
    "Noise-Free Evaluation Comparison"
)

# Run for GAIL
update_notebook(
    '/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb',
    "gail_loop", # Placeholder for loop detection logic
    "def plot_intermediate_results",
    "Noise-Free Evaluation Comparison"
)
