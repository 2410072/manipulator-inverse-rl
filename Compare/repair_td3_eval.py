import json

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'

def repair_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Content to restore
    td3_eval_section = [
        "\n",
        "**Evaluation vs Expert:**\n",
        "Finally, we compare the trained apprentices against the Expert baseline in a separate evaluation phase. The **performance comparison** below shows that the apprentices not only match but in some runs slightly exceed the expert's mean return, likely due to optimization on the simpler linearized reward landscape.\n",
        "*Specific Observation:* The expert baseline (dashed line) sits at approximately **-1.93**. The apprentices achieve comparable mean returns, with distribution spreads fully overlapping the expert's performance, validating that the projection method successfully recovered the expert's optimality.\n",
        "\n",
        "![TD3 Evaluation Comparison - Performance](Results/TD3/TD3_Evaluation_Comparison_Performance.png)\n",
        "*Figure 4: Final evaluation performance comparison (Apprentices vs Expert).*\n",
        "\n",
        "The **evaluation success rate** further corroborates this finding. All agents achieve near-perfect success rates, indistinguishable from the expert under stochastic evaluation conditions, validating the feature matching approach.\n",
        "*Specific Observation:* All agents achieve a success rate of **>98%** over the evaluation episodes, proving that the learned policy is robust to initialization noise.\n",
        "\n",
        "![TD3 Evaluation Comparison - Success Rate](Results/TD3/TD3_Evaluation_Comparison_SuccessRate.png)\n",
        "*Figure 5: Final evaluation success rate comparison (Apprentices vs Expert).*\n",
        "\n",
        "The **Evaluation Raster** provides a microscopic view of these testing episodes. The uniform yellow density across all apprentice runs confirms that the policies are not just successful on average, but consistently reliable across varied initialization states, exhibiting no signs of significant failure modes. This visual density is a hallmark of the stationary reward recovered by Projection IRL.\n",
        "\n",
        "![TD3 Evaluation Comparison - Raster](Results/TD3/TD3_Evaluation_Comparison_Raster.png)\n",
        "*Figure 6: Binary Success Raster during TD3 Evaluation.*\n"
    ]

    for cell in nb['cells']:
        source_str = "".join(cell['source'])
        
        # Locate the cell where the cut-off happened (it ends with the Raster Intro now)
        if "**Interpretation of Raster Plots:**" in source_str and "Figure 3: Binary Success Raster" in source_str:
            # Check if Evaluation section is already there (it shouldn't be based on my view)
            if "Evaluation vs Expert:" not in source_str:
                cell['source'] = cell['source'] + td3_eval_section
                print("Restored TD3 Evaluation Section.")
            else:
                print("TD3 Evaluation Section seems present already.")

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully repaired {NOTEBOOK_PATH}")

if __name__ == "__main__":
    repair_notebook()
