import json
import re

MD_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.md'
NB_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'

def is_image_line(line):
    return line.strip().startswith('![') and '](' in line and line.strip().endswith(')')

def cleanup_md():
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    prev_line = None
    
    removed_count = 0
    for line in lines:
        if is_image_line(line) and prev_line and line.strip() == prev_line.strip():
            removed_count += 1
            continue
        new_lines.append(line)
        if line.strip(): # Only update prev_line if not empty, or typically just update.
            # Actually, strict consecutive check including whitespace matches what I saw
            prev_line = line
            
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        
    print(f"Cleaned {MD_PATH}: Removed {removed_count} duplicate lines.")

def cleanup_nb():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    total_removed = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            new_source = []
            prev_line = None
            
            for line in source:
                # NB source lines often end with \n, so strip for comparison
                if is_image_line(line) and prev_line and line.strip() == prev_line.strip():
                    total_removed += 1
                    continue
                new_source.append(line)
                prev_line = line
            
            cell['source'] = new_source
            
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Cleaned {NB_PATH}: Removed {total_removed} duplicate lines.")

if __name__ == '__main__':
    cleanup_md()
    cleanup_nb()
