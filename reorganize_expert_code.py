import json

notebook_path = "TD3/Panda_Reach_v3_TD3_IRL.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            
            # Target Cell 9: Expert Evaluation
            if "TD3 Expert 直後評価" in source_text and "plot_intermediate_results" not in source_text:
                print(f"Found Expert Eval cell at index {i}")
                
                # Check formatting of the cell to append the plot command safely
                new_source = cell['source']
                
                # We want to add plotting logic at the end of this cell
                plot_code = [
                    "\n",
                    "# --- Plot Expert Baseline Immediately ---\n",
                    "# Use the 'successes' list populated above as the expert_history\n",
                    "print(\"Plotting Expert Baseline...\")\n",
                    "try:\n",
                    "    # We pass empty list for all_results since Apprentice hasn't run yet\n",
                    "    plot_intermediate_results(expert_history=successes, all_results=[], n_episodes=500)\n",
                    "except Exception as e:\n",
                    "    print(f\"Expert plotting failed: {e}\")\n"
                ]
                
                new_source.extend(plot_code)
                cell['source'] = new_source
                changed = True
                print("Added Expert plotting code to Cell 9")

    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {notebook_path}")
    else:
        print(f"No changes made. Plotting code might already exist or target cell not found.")

except Exception as e:
    print(f"Error: {e}")
