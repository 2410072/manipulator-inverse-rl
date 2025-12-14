
import json

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'
MARKDOWN_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.md'

def verify_sync():
    # Read Notebook content
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    nb_content = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            nb_content.append("".join(cell['source']))
            nb_content.append("\n\n")
    
    nb_full_text = "".join(nb_content).strip()

    # Read Markdown file content
    with open(MARKDOWN_PATH, 'r', encoding='utf-8') as f:
        md_full_text = f.read().strip()

    # Normalize newlines just in case
    nb_full_text = nb_full_text.replace("\r\n", "\n")
    md_full_text = md_full_text.replace("\r\n", "\n")

    if nb_full_text == md_full_text:
        print("VERIFICATION SUCCESS: Contents are identical.")
    else:
        print("VERIFICATION FAILED: Contents differ.")
        # Debug info
        print(f"Lengths: Notebook={len(nb_full_text)}, Markdown={len(md_full_text)}")
        # Simple diff?
        import difflib
        diff = difflib.unified_diff(
            nb_full_text.splitlines(), 
            md_full_text.splitlines(), 
            fromfile='Notebook', 
            tofile='Markdown', 
            lineterm=''
        )
        print("First few lines of diff:")
        for i, line in enumerate(diff):
            if i > 10: break
            print(line)

if __name__ == "__main__":
    verify_sync()
