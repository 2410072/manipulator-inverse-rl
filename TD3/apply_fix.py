import json
import os

notebook_path = 'Panda_Reach_v3_TD3_IRL.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if any('def projection_method_algorithm' in line for line in source):
                print("Found target cell.")
                
                # Check if already fixed
                if any('for i in range(max_runs):' in line for line in source):
                    print("Already fixed.")
                    found = True # Mark as found so we don't say "not found"
                    break

                insert_idx = -1
                for i, line in enumerate(source):
                    if 'if i == 0:' in line and '        ' in line: # Try to match the indentation of the if statement
                         # We want to insert BEFORE this block but AFTER local vars
                         pass
                    
                # Let's search for the line "        # Step 1: initialization for the very first apprentice"
                for i, line in enumerate(source):
                     if 'Step 1: initialization for the very first apprentice' in line:
                         insert_idx = i
                         break
                
                if insert_idx != -1:
                    new_lines = [
                        "\n",
                        "    for i in range(max_runs):\n",
                        "        apprentice = TD3Trainer(env=env, input_dims=obs_shape, agent_name=f'Apprentice_{i}', model_save_path=f'./Models/Apprentice_{i}/', exploration_period=exploration_period)\n",
                        "\n"
                    ]
                    source[insert_idx:insert_idx] = new_lines
                    print("Inserted lines.")
                    found = True
                    break
                else:
                    print("Target line for insertion not found.")

    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Saved notebook.")
    else:
        print("Could not find the target cell.")

except Exception as e:
    print(f"Error: {e}")
