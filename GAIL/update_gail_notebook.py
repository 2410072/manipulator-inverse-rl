import json

def update_notebook():
    notebook_path = "/home/shimoiyusuke/manipulator-inverse-rl/GAIL/Panda_Reach_v3_GAIL.ipynb"
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
        return

    # Identify the cell by content (approximate)
    target_string = "phase_name = \"Apprentice\""
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_content = "".join(cell['source'])
            if target_string in source_content and "gail_train" in source_content:
                # We found the training cell. We will replace it entirely with the looped version.
                
                new_source_lines = [
                    "# Automate training for Apprentice 0 to 10 (11 independent runs)\n",
                    "for i in range(11):\n",
                    "    print(f\"\\n--- Training Apprentice_{i} ---\\n\")\n",
                    "    phase_name = f\"Apprentice_{i}\"\n",
                    "    current_save_path = f\"./Models/Apprentice_{i}/\"\n",
                    "    n_train_episodes = 500\n",
                    "\n",
                    "    # Update kwargs for training\n",
                    "    current_kwargs = gail_agent_kwargs.copy()\n",
                    "    current_kwargs['batch_size'] = 256\n",
                    "    current_kwargs['exploration_period'] = 125\n",
                    "    current_kwargs['model_load_path'] = None\n",
                    "\n",
                    "    agent = GAILTrainer(**current_kwargs)\n",
                    "    agent.agent_name = phase_name\n",
                    "    agent.model_save_path = current_save_path\n",
                    "\n",
                    "    # Train\n",
                    "    score_history, avg_score_history, success_history, avg_success_history = agent.gail_train(\n",
                    "        n_episodes=n_train_episodes,\n",
                    "        opt_steps=50,\n",
                    "        print_every=50,\n",
                    "        plot_save_path=f'../Results/GAIL/{phase_name}_Performance.png'\n",
                    "    )\n",
                    "\n",
                    "    agent.save_model()\n",
                    "    print(f\"Training complete for {phase_name}. Model saved to {current_save_path}\")\n"
                ]
                
                cell['source'] = new_source_lines
                modified = True
                print("Successfully modified GAIL notebook training cell.")
                break

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Notebook {notebook_path} updated.")
    else:
        print("Target cell not found in GAIL notebook.")

if __name__ == "__main__":
    update_notebook()
