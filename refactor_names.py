import os
import shutil

def replace_in_file(filepath, replacements):
    """
    Reads a file, applies string replacements, and writes it back.
    replacements: dict of {old_string: new_string}
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping replacement.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes needed in {filepath}")

def refactor_irl():
    print("--- Refactoring IRL Directory ---")
    base_dir = "IRL"
    
    # 1. Rename td3_algo.py -> irl_algo.py
    old_algo = os.path.join(base_dir, "td3_algo.py")
    new_algo = os.path.join(base_dir, "irl_algo.py")
    
    if os.path.exists(old_algo):
        shutil.move(old_algo, new_algo)
        print(f"Renamed {old_algo} -> {new_algo}")
    elif not os.path.exists(new_algo):
        print(f"Error: {old_algo} not found!")
    else:
        print(f"{new_algo} already exists.")

    # 2. Update content in irl_algo.py
    # Replacing TD3->IRL, td3->irl (careful with 'td3' if it's part of other words, but usually safe in this context)
    algo_replacements = {
        "class TD3Trainer": "class IRLTrainer",
        "def td3_train": "def irl_train",
        "TD3": "IRL", # General replacement for docs/comments
        "td3_": "irl_" # Prefix replacement
    }
    replace_in_file(new_algo, algo_replacements)

    # 3. Update Notebook
    notebook_path = os.path.join(base_dir, "Panda_Reach_v3_IRL.ipynb")
    notebook_replacements = {
        "from td3_algo import TD3Trainer": "from irl_algo import IRLTrainer",
        "expert = TD3Trainer": "expert = IRLTrainer", # If expert uses it
        "apprentice = TD3Trainer": "apprentice = IRLTrainer",
        ".td3_train": ".irl_train",
        "TD3": "IRL", # Visual text update
        # Fix paths if they were hardcoded with TD3
        "/TD3/": "/IRL/", 
        "./Models/TD3/": "./Models/IRL/"
    }
    replace_in_file(notebook_path, notebook_replacements)

def refactor_gail():
    print("\n--- Refactoring GAIL Directory ---")
    base_dir = "GAIL"
    
    # GAIL might use TD3 as a base or have copy-pasted comments.
    # User said "Apply that change to GAIL", implying consistent naming (GAIL for GAIL directory).
    
    # 1. Check gail_algo.py
    algo_path = os.path.join(base_dir, "gail_algo.py")
    # If GAIL is implemented based on TD3 (common), we rename the class to GAILTrainer if it isn't already.
    # Or if it calls 'td3_train', we rename to 'gail_train'.
    algo_replacements = {
        "class TD3Trainer": "class GAILTrainer",
        "def td3_train": "def gail_train",
        "TD3": "GAIL" # General text update
    }
    replace_in_file(algo_path, algo_replacements)

    # 2. Check imports in gail_algo.py if it imports from sibling files?
    # (Assuming it's self contained or imports util)

    # 3. Update Notebook
    notebook_path = os.path.join(base_dir, "Panda_Reach_v3_GAIL.ipynb")
    notebook_replacements = {
        "TD3Trainer": "GAILTrainer",
        "td3_train": "gail_train",
        "TD3": "GAIL" # Visual text
    }
    replace_in_file(notebook_path, notebook_replacements)

if __name__ == "__main__":
    refactor_irl()
    refactor_gail()
    print("\nRefactoring complete.")
