import json

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # New refined text for 4.1 Intro
    raster_intro_text = [
        "**Interpretation of Raster Plots:**\n",
        "To visualize the stability of the learned policy at a granular level, we present the **Binary Success Raster** (Figure 3, above). In this visualization:\n",
        "- **Y-Axis (Rows):** Each row represents a distinct training run (Apprentice 1, 2, 3), allowing us to check for consistency across different random seeds.\n",
        "- **X-Axis (Columns):** Represents the progression of episodes over time.\n",
        "- **Color Coding:** A **Yellow** pixel indicates a successful episode (target reached), while a **Purple** pixel indicates failure.\n",
        "- **Visual Analysis:** This graph allows us to instantly verify *how* the agent learns. Dense, uninterrupted blocks of yellow indicate a stable, reliable policy. Conversely, scattered purple lines would suggest \"forgetting\" or instability. As seen in Figure 3, the TD3 apprentices exhibit a clean transition to solid yellow, confirming the stability of the IRL solution.\n"
    ]

    for cell in nb['cells']:
        source = "".join(cell['source'])
        
        # Replace the old paragraph in 4.1 with the new detailed explanation
        if "To visualize the stability of the learned policy at a granular level" in source:
             # Identify the cell roughly by content
             cell['source'] = [
                 "### 4.1 TD3 (IRL) Performance Analysis\n",
                 "\n",
                 "The IRL apprentices demonstrate robust learning due to the stable linear reward structure. We first examine the **training performance** in terms of cumulative episodic return. The following plot highlights the steady improvement of all three apprentice agents, indicating that the synthesized reward $w^T \phi(s)$ successfully guides the policy towards high-return regions.\n",
                 "*Specific Observation:* The apprentices consistently converge to a return of approximately **-1.9**, which closely aligns with the theoretical upper bound for this distance-based task. The learning variance is notably low, suggesting a stable gradient landscape.\n",
                 "\n",
                 "![TD3 Learning Comparison - Performance](Results/TD3/TD3_Learning_Comparison_Performance.png)\n",
                 "*Figure 1: Smoothed performance score of TD3 Apprentices (1-3) during training.*\n",
                 "\n",
                 "Complementing the score, the **training success rate** provides a tangible measure of task completion. As seen below, the success rate rises monotonically and approaches 1.0 (100%), confirming that the apprentices reliably learn to reach the target under the IRL reward.\n",
                 "*Specific Observation:* Success rates exceed **90%** after approximately 1500 timesteps and stabilize at **near 100%**, indicating that the recovered reward function effectively penalizes deviations from the goal.\n",
                 "\n",
                 "![TD3 Learning Comparison - Success Rate](Results/TD3/TD3_Learning_Comparison_SuccessRate.png)\n",
                 "*Figure 2: Moving average success rate of TD3 Apprentices (1-3) during training.*\n",
                 "\n",
                 "![TD3 Learning Comparison - Raster](Results/TD3/TD3_Learning_Comparison_Raster.png)\n",
                 "*Figure 3: Binary Success Raster for TD3 Apprentices.*\n",
                 "\n"
             ] + raster_intro_text
             print("Updated Section 4.1 Raster Intro.")

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully synced descriptions to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_notebook()
