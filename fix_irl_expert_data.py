import json

nb_path = '/home/shimoiyusuke/manipulator-inverse-rl/IRL/Panda_Reach_v3_IRL.ipynb'

print(f"Fixing Expert Data in {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
changes = 0

target_str = "{'name': 'Expert', 'scores': [], 'successes': []}, # Placeholder if collected"
replace_str = "{'name': 'Expert', 'scores': score_history, 'successes': success_history},"

for cell in cells:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if target_str in source:
            source = source.replace(target_str, replace_str)
            cell['source'] = [line + "\n" for line in source.splitlines()]
            # Fix double newlines locally
            cell['source'] = [l.replace('\n\n', '\n') for l in cell['source']]
            changes += 1
            print("Found and replaced target string.")

if changes > 0:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Saved {nb_path} with {changes} changes.")
else:
    print("Target string not found. No changes made.")
