import json
import os

def fix_irl_notebook():
    nb_path = "/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb"
    print(f"Reading {nb_path}...")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            cell_modified = False
            
            # Check if this cell contains gym.make for PandaReach-v3
            full_source = "".join(source)
            if "gym.make" in full_source and "PandaReach-v3" in full_source:
                print("Found gym.make cell.")
                
                for line in source:
                    # Target lines to comment out
                    targets = [
                        'renderer="OpenGL"',
                        'render_target_position=',
                        'render_distance=',
                        'render_yaw=',
                        'render_pitch='
                    ]
                    
                    should_comment = False
                    for t in targets:
                        # Check strictly if it's not already commented
                        if t in line and not line.strip().startswith("#"):
                            should_comment = True
                            break
                    
                    if should_comment:
                        print(f"  Commenting out: {line.strip()}")
                        # Preserve indentation but add #
                        # Assuming 4 spaces indent usually, we just add # at start of non-whitespace? 
                        # Or just replace the line content.
                        # Let's simple insert # after the leading whitespace
                        ws_len = len(line) - len(line.lstrip())
                        ws = line[:ws_len]
                        content = line[ws_len:]
                        new_line = f"{ws}# {content}"
                        new_source.append(new_line)
                        cell_modified = True
                    else:
                        new_source.append(line)
                
                if cell_modified:
                    cell['source'] = new_source
                    modified = True
                    print("  Cell updated.")
            
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook saved successfully.")
    else:
        print("No changes were needed or target code not found.")

if __name__ == "__main__":
    fix_irl_notebook()
