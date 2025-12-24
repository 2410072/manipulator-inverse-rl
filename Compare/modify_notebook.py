
import json
import os
import sys

NOTEBOOK_PATH = "/home/shimoiyusuke/manipulator-inverse-rl/Compare/Compare.ipynb"

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def main():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    td3_found = False
    gail_found = False

    for cell in nb['cells']:
        source_str = "".join(cell.get('source', []))
        
        # 1. Replace TD3 monolithic cell
        if "td3_runner.train_apprentices" in source_str and not td3_found:
            print("Found TD3 training cell. Replacing with granular cells...")
            td3_found = True
            
            # --- Cell 1: Setup & Expert Features ---
            new_cells.append(create_code_cell([
                "# Train TD3 Apprentices\n",
                "from irl_utils import create_env, get_obs_shape\n",
                "print(\"\\n--- Training TD3 Apprentices ---\")\n",
                "\n",
                "env = create_env()\n",
                "obs_shape = get_obs_shape(env)\n",
                "\n",
                "# 1. Compute Expert Features\n",
                "print(\"Step 1: Computing Expert Features\")\n",
                "expert_feature_expectation, _ = td3_runner.compute_expert_features(expert, env, m=config.FEATURE_EXPECTATION_EPISODES)\n",
                "\n",
                "feature_expectation = []\n",
                "feature_expectation_bar = []\n",
                "all_td3_results = []"
            ]))
            
            # --- Cell 2: Apprentice 0 ---
            new_cells.append(create_code_cell([
                "# 2. Train Apprentice 0\n",
                "print(\"Step 2: Training Apprentice 0\")\n",
                "# Use config.SEED for reproducibility or None for random inside the function\n",
                "res0, feat0, w0, m0 = td3_runner.train_apprentice_0(env, obs_shape, seed=config.SEED)\n",
                "all_td3_results.append(res0)\n",
                "feature_expectation.append(feat0)"
            ]))
            
            # --- Cell 3: Apprentice 1 ---
            new_cells.append(create_code_cell([
                "# 3. Train Apprentice 1\n",
                "i = 1\n",
                "print(f\"Step 3.{i}: Training Apprentice {i}\")\n",
                "res, feat, w, m, new_bar, converged = td3_runner.train_apprentice_i(\n",
                "    i, env, obs_shape, expert_feature_expectation, \n",
                "    feature_expectation, feature_expectation_bar, seed=config.SEED\n",
                ")\n",
                "\n",
                "if new_bar is not None:\n",
                "    feature_expectation_bar.append(new_bar)\n",
                "\n",
                "all_td3_results.append(res)\n",
                "feature_expectation.append(feat)"
            ]))
            
            # --- Cell 4: Apprentice 2 ---
            new_cells.append(create_code_cell([
                "# 3. Train Apprentice 2\n",
                "i = 2\n",
                "print(f\"Step 3.{i}: Training Apprentice {i}\")\n",
                "res, feat, w, m, new_bar, converged = td3_runner.train_apprentice_i(\n",
                "    i, env, obs_shape, expert_feature_expectation, \n",
                "    feature_expectation, feature_expectation_bar, seed=config.SEED\n",
                ")\n",
                "\n",
                "if new_bar is not None:\n",
                "    feature_expectation_bar.append(new_bar)\n",
                "\n",
                "all_td3_results.append(res)\n",
                "feature_expectation.append(feat)"
            ]))
            
            # --- Cell 5: Apprentice 3 ---
            new_cells.append(create_code_cell([
                "# 3. Train Apprentice 3\n",
                "i = 3\n",
                "print(f\"Step 3.{i}: Training Apprentice {i}\")\n",
                "res, feat, w, m, new_bar, converged = td3_runner.train_apprentice_i(\n",
                "    i, env, obs_shape, expert_feature_expectation, \n",
                "    feature_expectation, feature_expectation_bar, seed=config.SEED\n",
                ")\n",
                "\n",
                "if new_bar is not None:\n",
                "    feature_expectation_bar.append(new_bar)\n",
                "\n",
                "all_td3_results.append(res)\n",
                "feature_expectation.append(feat)"
            ]))
            
             # --- Cell 6: Collect Results ---
            new_cells.append(create_code_cell([
                "td3_results = all_td3_results\n",
                "# Plot apprentice comparison after all training is complete\n",
                "from plotting import plot_apprentice_comparison\n",
                "plot_apprentice_comparison(\n",
                "    \"TD3\",\n",
                "    all_td3_results,\n",
                "    save_dir=config.TD3_RESULTS_DIR\n",
                ")"
            ]))

        # 2. Replace GAIL monolithic cell
        elif "gail_runner.train_apprentices" in source_str and not gail_found and "def train_apprentices" not in source_str:
            print("Found GAIL training cell. Replacing with granular cells...")
            gail_found = True
            
            # --- Cell 1: Setup ---
            new_cells.append(create_code_cell([
                "# Train GAIL Apprentices\n",
                "print(\"\\n--- Training GAIL Apprentices ---\")\n",
                "\n",
                "# Ensure expert trajectories exist\n",
                "gail_runner.ensure_expert_trajectories()\n",
                "\n",
                "all_gail_results = []"
            ]))
            
             # --- Cell 2: Apprentice 1 ---
            new_cells.append(create_code_cell([
                "# Train GAIL Apprentice 1\n",
                "i = 1\n",
                "print(f\"Step 4.{i}: Training GAIL Apprentice {i}\")\n",
                "res = gail_runner.train_gail_apprentice_i(i, seed=config.SEED)\n",
                "all_gail_results.append(res)"
            ]))
            
            # --- Cell 3: Apprentice 2 ---
            new_cells.append(create_code_cell([
                "# Train GAIL Apprentice 2\n",
                "i = 2\n",
                "print(f\"Step 4.{i}: Training GAIL Apprentice {i}\")\n",
                "res = gail_runner.train_gail_apprentice_i(i, seed=config.SEED)\n",
                "all_gail_results.append(res)"
            ]))
            
            # --- Cell 4: Apprentice 3 ---
            new_cells.append(create_code_cell([
                "# Train GAIL Apprentice 3\n",
                "i = 3\n",
                "print(f\"Step 4.{i}: Training GAIL Apprentice {i}\")\n",
                "res = gail_runner.train_gail_apprentice_i(i, seed=config.SEED)\n",
                "all_gail_results.append(res)"
            ]))
            
            # --- Cell 5: Collect Results ---
            new_cells.append(create_code_cell([
                "gail_results = all_gail_results\n",
                "# Plot apprentice comparison after all training is complete\n",
                "from plotting import plot_apprentice_comparison\n",
                "plot_apprentice_comparison(\n",
                "    \"GAIL\",\n",
                "    all_gail_results,\n",
                "    save_dir=config.GAIL_RESULTS_DIR\n",
                ")"
            ]))
            
        else:
            new_cells.append(cell)

    if not td3_found:
        print("Warning: TD3 training cell not found!")
    if not gail_found:
        print("Warning: GAIL training cell not found!")

    nb['cells'] = new_cells
    
    print(f"Writing modified notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Done.")

if __name__ == "__main__":
    main()
