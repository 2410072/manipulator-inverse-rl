import json

notebooks = [
    {
        "path": "TD3/Panda_Reach_v3_TD3_IRL.ipynb",
        "target_string": "exploration_period=50",
        "replacement_string": "exploration_period=100"
    },
    {
        "path": "GAIL/Panda_Reach_v3_GAIL.ipynb",
        "target_string": "exploration_period=50",
        "replacement_string": "exploration_period=100"
    }
]

for nb_info in notebooks:
    path = nb_info["path"]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        changed = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    if nb_info["target_string"] in line:
                        new_line = line.replace(nb_info["target_string"], nb_info["replacement_string"])
                        new_source.append(new_line)
                        changed = True
                    else:
                        new_source.append(line)
                cell['source'] = new_source
        
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Updated {path}")
        else:
            print(f"No changes made to {path} (Target not found)")

    except Exception as e:
        print(f"Error processing {path}: {e}")
