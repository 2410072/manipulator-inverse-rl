
import json
import re

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'

def remove_section():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    in_section_to_remove = False
    
    for cell in nb['cells']:
        source_str = "".join(cell['source'])
        
        # Check for start of 4.3
        if "### 4.3 Direct Comparison (TD3 vs GAIL)" in source_str:
            in_section_to_remove = True
            print("Found Section 4.3 start. Deleting...")
            continue # Skip this cell
            
        # Check for start of 4.4 (End of removal)
        if "### 4.4 Comparative Evaluation Analysis" in source_str:
            in_section_to_remove = False
            print("Found Section 4.4 start. Stopping deletion and Renumbering...")
            
            # Renumber Section 4.4 -> 4.3
            new_source = []
            for line in cell['source']:
                line = line.replace("### 4.4 Comparative Evaluation Analysis", "### 4.3 Comparative Evaluation Analysis")
                # Renumber (old) Figure 18 to Figure 12
                # Note: previously I updated it from 15 to 18. Now from 18 to 12.
                # To be safe, I'll regex replace `Figure \d+` if matching the context, or just hardcode if I text-match.
                if "Comparative Dashboard of Evaluation Results" in line:
                    line = re.sub(r"Figure \d+", "Figure 12", line)
                new_source.append(line)
            cell['source'] = new_source
            new_cells.append(cell)
            continue

        if in_section_to_remove:
            print("Skipping cell in deleted section...")
            continue
            
        new_cells.append(cell)

    nb['cells'] = new_cells
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully updated {NOTEBOOK_PATH}")

if __name__ == "__main__":
    remove_section()
