
import nbformat
import re

notebook_path = '/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

fixed = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Need to match the block with batch_size=1024
        if "current_kwargs['batch_size'] = 1024" in cell.source:
            
            # Replace batch_size
            new_source = cell.source.replace(
                "current_kwargs['batch_size'] = 1024 # GPU Optimization",
                "current_kwargs['batch_size'] = 256 # Optimized for shorter episodes"
            )
            
            # Comment out model_load_path
            new_source = new_source.replace(
                "current_kwargs['model_load_path'] = load_path",
                "# current_kwargs['model_load_path'] = load_path # Disabled Warm Start"
            )
            
            # Comment out exploration reduction logic
            # Using regex for multiline replace or simpler string replace if exact
            # The structure is:
            # if run_idx > 0:
            #     base_expl = current_kwargs.get('exploration_period', 100)
            #     current_kwargs['exploration_period'] = int(base_expl * 0.5)
            #     print(f"Warm Start: Loading model from {load_path}")
            
            block_to_replace = """    if run_idx > 0:
        base_expl = current_kwargs.get('exploration_period', 100)
        current_kwargs['exploration_period'] = int(base_expl * 0.5)
        print(f"Warm Start: Loading model from {load_path}")"""

            replacement_block = """    if run_idx > 0:
        # base_expl = current_kwargs.get('exploration_period', 100)
        # current_kwargs['exploration_period'] = int(base_expl * 0.5)
        # print(f"Warm Start: Loading model from {load_path}")
        print("Warm Start: Disabled (Fresh Start)")"""
            
            if block_to_replace in new_source:
                new_source = new_source.replace(block_to_replace, replacement_block)
                cell.source = new_source
                fixed = True
                print("Fixed GAIL notebook batch size and warm start logic.")
            else:
                 print("Could not find the exact warm start block to replace.")

if fixed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
else:
    print("Could not find the target code block in GAIL notebook.")
