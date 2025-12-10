import json

notebook_path = "TD3/Panda_Reach_v3_TD3_IRL.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            
            if "def projection_method_algorithm" in source_text or "projection_method_algorithm(" in source_text:
                new_source = []
                for line in cell['source']:
                    # Update Warm Start logic first (order matters to avoid double replace issues if not careful, though specific strings help)
                    # Old: load_path = f'./Models/Apprentice {i-1}/' if i > 0 else None
                    if "load_path = f'./Models/Apprentice {i-1}/'" in line:
                        new_line = line.replace("{i-1}", "{i}")
                        new_source.append(new_line)
                        changed = True
                    
                    # Update agent creation
                    # Old: agent_name=f'Apprentice {i}'
                    elif "agent_name=f'Apprentice {i}'" in line:
                        new_line = line.replace("{i}", "{i+1}")
                        new_source.append(new_line)
                        changed = True
                    # Old: model_save_path=f'./Models/Apprentice {i}/'
                    elif "model_save_path=f'./Models/Apprentice {i}/'" in line:
                        new_line = line.replace("{i}", "{i+1}")
                        new_source.append(new_line)
                        changed = True
                    
                    # Update plot save path
                    # Old: plot_save_path=f'../Results/TD3/Apprentice_{i}_Performance.png'
                    elif "plot_save_path=f'../Results/TD3/Apprentice_{i}_Performance.png'" in line:
                        new_line = line.replace("{i}", "{i+1}")
                        new_source.append(new_line)
                        changed = True
                        
                    # Update render save path
                    # Old: render_save_path=f'../Results/TD3/Apprentice {i} Policy'
                    elif "render_save_path=f'../Results/TD3/Apprentice {i} Policy'" in line:
                        new_line = line.replace("{i}", "{i+1}")
                        new_source.append(new_line)
                        changed = True

                    # Update Results ID
                    # Old: "id": i,
                    elif "\"id\": i," in line:
                        new_line = line.replace("i,", "i+1,")
                        new_source.append(new_line)
                        changed = True
                        
                    # Update Print Statements
                    # Old: print(f"Apprentice {i} Final Avg Success Rate: {final_sr:.1f}%")
                    elif "Apprentice {i} Final Avg Success Rate" in line:
                        new_line = line.replace("{i}", "{i+1}")
                        new_source.append(new_line)
                        changed = True
                    # Old: print(f"\nDetailed Stats for Apprentice {i}:")
                    elif "Detailed Stats for Apprentice {i}" in line:
                        new_line = line.replace("{i}", "{i+1}")
                        new_source.append(new_line)
                        changed = True
                    
                    else:
                        new_source.append(line)
                cell['source'] = new_source

    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {notebook_path}")
    else:
        print("No changes made. Target strings not found.")

except Exception as e:
    print(f"Error: {e}")
