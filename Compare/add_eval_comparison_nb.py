import json

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # New content blocks for each apprentice
    apprentice_1_eval = [
        "\n",
        "**Evaluation Phase (Apprentice 1):**\n",
        "Comparing the deployed policies in a stochastic evaluation setting (50 episodes), both agents achieve high success rates. The IRL policy (TD3) exhibits a dense and unbroken success raster, indicating robust generalization. GAIL matches the performance level but may show minor discontinuities in the raster, a footprint of the adversarial training's mode-seeking behavior.\n",
        "\n",
        "![TD3 vs GAIL - Apprentice 1 (Evaluation)](Results/TD3_vs_GAIL_Evaluation_Apprentice_1.png)\n",
        "*Figure 13: Evaluation comparison for Apprentice 1 (TD3 vs GAIL).*\n"
    ]

    apprentice_2_eval = [
        "\n",
        "**Evaluation Phase (Apprentice 2):**\n",
        "For the second apprentice, the evaluation results confirm the training trends. TD3-IRL provides a stable, low-variance return profile. The GAIL agent is highly competitive in terms of peak performance, often indistinguishable from TD3 in success rate, though the raster comparison highlights the superior stability of the projection-based approach.\n",
        "\n",
        "![TD3 vs GAIL - Apprentice 2 (Evaluation)](Results/TD3_vs_GAIL_Evaluation_Apprentice_2.png)\n",
        "*Figure 15: Evaluation comparison for Apprentice 2 (TD3 vs GAIL).*\n"
    ]

    apprentice_3_eval = [
        "\n",
        "**Evaluation Phase (Apprentice 3):**\n",
        "The final apprentice run underscores the reliability difference. While both methods solve the task, the TD3-IRL evaluation raster is uniformly successful across the episode batch. GAIL achieves similar mean returns but with slightly higher variance in the success distribution, validating the hypothesis that stationary rewards lead to more robust final policies.\n",
        "\n",
        "![TD3 vs GAIL - Apprentice 3 (Evaluation)](Results/TD3_vs_GAIL_Evaluation_Apprentice_3.png)\n",
        "*Figure 17: Evaluation comparison for Apprentice 3 (TD3 vs GAIL).*\n"
    ]

    for cell in nb['cells']:
        source = "".join(cell['source'])
        
        # Modify Section 4.3 cell (assuming it's all in one cell or we find the parts)
        # Check if it's the 4.3 cell
        if "### 4.3 Direct Comparison (TD3 vs GAIL)" in source:
            new_source = []
            lines = cell['source']
            i = 0
            while i < len(lines):
                line = lines[i]
                new_source.append(line)
                
                # Insert Appr 1 Eval
                if "*Figure 12: Direct comparison for Apprentice 1 (TD3 vs GAIL).*" in line:
                    if "**Evaluation Phase (Apprentice 1):**" not in source: # Avoid double insert
                        new_source.extend(apprentice_1_eval)
                
                # Insert Appr 2 Eval AND Update Figure Number if needed (Figure 13 -> 14)
                if "*Figure 13: Direct comparison for Apprentice 2 (TD3 vs GAIL).*" in line:
                    new_source[-1] = line.replace("Figure 13", "Figure 14")
                    if "**Evaluation Phase (Apprentice 2):**" not in source:
                        new_source.extend(apprentice_2_eval)
                        
                # Insert Appr 3 Eval AND Update Figure Number if needed (Figure 14 -> 16)
                if "*Figure 14: Direct comparison for Apprentice 3 (TD3 vs GAIL).*" in line:
                    new_source[-1] = line.replace("Figure 14", "Figure 16")
                    if "**Evaluation Phase (Apprentice 3):**" not in source:
                        new_source.extend(apprentice_3_eval)
                
                i += 1
            cell['source'] = new_source
            print("Updated Section 4.3 with Evaluation Comparison.")

        # Update Section 4.4 Figure Number (Figure 15 -> 18)
        if "*Figure 15: Comparative Dashboard of Evaluation Results (TD3 vs GAIL).*" in source:
             cell['source'] = [l.replace("Figure 15", "Figure 18") for l in cell['source']]
             print("Updated Section 4.4 Figure Number.")

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully updated {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_notebook()
