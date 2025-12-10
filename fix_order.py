import json

notebook_path = "TD3/Panda_Reach_v3_TD3_IRL.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    plot_def_index = -1
    expert_eval_index = -1
    
    # Locate cells
    for i, cell in enumerate(nb['cells']):
        source_text = "".join(cell['source'])
        
        if "def plot_intermediate_results" in source_text:
            plot_def_index = i
            print(f"Found plot_intermediate_results definition at index {i}")
        
        if "TD3 Expert 直後評価" in source_text:
            expert_eval_index = i
            print(f"Found Expert Eval at index {i}")

    # Reorder if necessary
    if plot_def_index != -1 and expert_eval_index != -1:
        if plot_def_index > expert_eval_index:
            print("Definition is AFTER Call. Moving definition cell...")
            
            # Pop the definition cell
            plot_cell = nb['cells'].pop(plot_def_index)
            
            # Insert it before the eval cell
            # Note: After popping, indices might shift if plot_def_index < expert_eval_index, 
            # but here plot_def_index > expert_eval_index, so expert_eval_index remains valid.
            nb['cells'].insert(expert_eval_index, plot_cell)
            
            print(f"Moved plot definition from {plot_def_index} to {expert_eval_index}")
            
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Successfully updated {notebook_path}")
        else:
            print("Order is correct (Definition is before Call).")
            # If order is correct but user says "failed", maybe we need to ensure the cell *was run*.
            # But we can only fix the static order here.
    else:
        print("Could not find one of the target cells.")

except Exception as e:
    print(f"Error: {e}")
