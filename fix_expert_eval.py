import json

notebook_path = "TD3/Panda_Reach_v3_TD3_IRL.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Fix 1: Update td3_train unpacking to handle 4 values
                if "score_history, avg_score_history = expert.td3_train" in line:
                    new_line = line.replace(
                        "score_history, avg_score_history = expert.td3_train",
                        "score_history, avg_score_history, success_history, avg_success_history = expert.td3_train"
                    )
                    new_source.append(new_line)
                    changed = True
                    print("Fixed td3_train unpacking")
                
                # Fix 2: Update Expert Evaluation loop (lightweight eval) to 500 episodes
                elif "for _ in range(10):" in line:
                    # Check context: verify this is the evaluation cell (contains expert.test_model)
                    is_eval_cell = any("expert.test_model" in l for l in cell['source'])
                    if is_eval_cell:
                        new_line = line.replace("range(10)", "range(500)")
                        new_source.append(new_line)
                        changed = True
                        print("Updated Expert evaluation loop to 500 episodes")
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
    
    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {notebook_path}")
    else:
        print(f"No changes needed for {notebook_path}")

except Exception as e:
    print(f"Error: {e}")
