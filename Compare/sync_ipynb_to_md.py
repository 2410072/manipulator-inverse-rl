
import json
import os

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'
MARKDOWN_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.md'

def sync_notebook_to_markdown():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    markdown_content = []
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            # Join the list of strings in 'source'
            source = "".join(cell['source'])
            markdown_content.append(source)
            # Add a couple of newlines between cells to mimic standard markdown separation
            markdown_content.append("\n\n")

    # Join all cells
    full_content = "".join(markdown_content).strip() + "\n"

    print(f"Writing content to {MARKDOWN_PATH}...")
    with open(MARKDOWN_PATH, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print("Sync complete.")

if __name__ == "__main__":
    sync_notebook_to_markdown()
