import json

notebook_path = "TD3/Panda_Reach_v3_TD3_IRL.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    expert_eval_index = -1
    plot_def_source = []
    
    # 1. Locate "TD3 Expert 直後評価" (Cell 9 / Usage)
    for i, cell in enumerate(nb['cells']):
        source_text = "".join(cell['source'])
        if "TD3 Expert 直後評価" in source_text:
            expert_eval_index = i
            print(f"Found Expert Eval (Usage) at index {i}")
            break
            
    if expert_eval_index == -1:
        print("Error: Could not find Expert Evaluation cell.")
        exit(1)

    # 2. Extract `plot_intermediate_results` definition content and remove ALL existing definition cells
    # We scan all cells. If we find the definition, we save it (if not already saved) and delete the cell.
    cells_to_keep = []
    
    for i, cell in enumerate(nb['cells']):
        source_text = "".join(cell['source'])
        if "def plot_intermediate_results" in source_text:
            print(f"Found definition at index {i}. Removing it to consolidate/move.")
            if not plot_def_source:
                plot_def_source = cell['source']
        else:
            cells_to_keep.append(cell)
            
    nb['cells'] = cells_to_keep
    
    if not plot_def_source:
        print("Error: Could not find `plot_intermediate_results` definition anywhere.")
        exit(1)

    # 3. Insert definition strictly BEFORE the Expert Eval cell
    # Note: indexes in `cells_to_keep` might have shifted relative to original `nb['cells']`
    # We need to find the Expert Eval cell again in the *new* list
    
    new_expert_eval_index = -1
    for i, cell in enumerate(nb['cells']):
        source_text = "".join(cell['source'])
        if "TD3 Expert 直後評価" in source_text:
            new_expert_eval_index = i
            break
            
    if new_expert_eval_index != -1:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "plot_def_fixed",
            "metadata": {},
            "outputs": [],
            "source": plot_def_source
        }
        nb['cells'].insert(new_expert_eval_index, new_cell)
        print(f"Inserted plot definition before Expert Eval (New Index: {new_expert_eval_index})")
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {notebook_path}")
    else:
        print("Error: Usage cell disappeared during cleanup? (Should not happen)")

except Exception as e:
    print(f"Error: {e}")
