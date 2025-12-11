import json

notebooks = [
    '/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb',
    '/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb'
]

target_line = "calculate_chunked_stats(success_history, chunk_size=50)"

for nb_path in notebooks:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        changes = 0
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    # Check if this is the line needing a comma
                    if target_line in line and not line.strip().endswith(","):
                        # Ensure it's inside the dictionary by context (next line has noise_free)
                        # But simplest is just to add comma if it looks like the function call ending
                        new_line = line.replace("\n", ",\n")
                        new_source.append(new_line)
                        changes += 1
                        print(f"Fixed missing comma in {nb_path}")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                
        if changes > 0:
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Saved {changes} fixes to {nb_path}")
        else:
             print(f"No syntax errors found/fixed in {nb_path}")
             
    except Exception as e:
        print(f"Error processing {nb_path}: {e}")
