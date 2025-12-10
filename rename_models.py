import os
import shutil

base_path = "TD3/Models"
# Range of apprentices found in git (0 to 15 based on ls-tree)
# We rename in reverse to avoid overwriting: 15->16, ..., 0->1

for i in range(15, -1, -1):
    old_name = f"Apprentice {i}"
    new_name = f"Apprentice {i+1}"
    
    old_path = os.path.join(base_path, old_name)
    new_path = os.path.join(base_path, new_name)
    
    if os.path.exists(old_path):
        if os.path.exists(new_path):
             print(f"Warning: Target {new_path} already exists. Skipping or handling collision.")
             # In a clean restore, this shouldn't happen unless 1->2 and 2 exists. 
             # Reverse order handles this.
        
        print(f"Renaming {old_name} -> {new_name}")
        shutil.move(old_path, new_path)
    else:
        print(f"Skipping {old_name} (Not found)")

# Clean up underscore versions if they exist and aren't wanted (user seemed to use space versions)
# But let's leave them if they were checked out, just in case. Or maybe the user wants them too?
# The notebook uses space. Focusing on space.
