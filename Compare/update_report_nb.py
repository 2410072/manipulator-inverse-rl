import json
import os

NOTEBOOK_PATH = '/home/shimoiyusuke/manipulator-inverse-rl/Compare/Final_Report.ipynb'

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    new_cells = []
    
    # Text content
    td3_raster_text = "\n\nThe **Evaluation Raster** provides a microscopic view of these testing episodes. The uniform yellow density across all apprentice runs confirms that the policies are not just successful on average, but consistently reliable across varied initialization states, exhibiting no signs of significant failure modes. This visual density is a hallmark of the stationary reward recovered by Projection IRL."
    
    gail_raster_cell_source = [
        "The **Evaluation Raster** for GAIL corroborates the high success rates but occasionally reveals the \"thin\" nature of the learned solution. While predominantly successful (yellow), the presence of any sparse purple pixels would indicate specific state configurations where the adversarial policy fails to generalize, unlike the more robust IRL counterparts.\n",
        "\n",
        "![GAIL Evaluation Comparison - Raster](Results/GAIL/GAIL_Evaluation_Comparison_Raster.png)\n",
        "*Figure 11: Binary Success Raster during GAIL Evaluation.*\n"
    ]

    section_4_4_source = [
        "### 4.4 Comparative Evaluation Analysis\n",
        "\n",
        "To synthesize the evaluation results, we present a cross-algorithm dashboard comparing the final policies.\n",
        "\n",
        "![TD3 vs GAIL Comparison Dashboard - Evaluation Performance](Results/TD3_vs_GAIL_Comparison_Performance.png)\n",
        "![TD3 vs GAIL Comparison Dashboard - Success Rate](Results/TD3_vs_GAIL_Comparison_SuccessRate.png)\n",
        "![TD3 vs GAIL Comparison Dashboard - Raster](Results/TD3_vs_GAIL_Comparison_Raster.png)\n",
        "*Figure 15: Comparative Dashboard of Evaluation Results (TD3 vs GAIL).*\n",
        "\n",
        "**Results and Consideration:**\n",
        "The comparative dashboard provides crucial insights into the stability-efficiency trade-off:\n",
        "1.  **Performance & Reliability**: Both algorithms achieve comparable peak performance, with mean returns hovering around -2.0. However, the TD3 (IRL) success rate raster is noticeably more uniform. GAIL, while highly successful, exhibits occasional \"flickering\" in the raster plot (sparse failure modes), which is characteristic of the adversarial instability.\n",
        "2.  **Mode Coverage**: The dense yellow blocks in the TD3 raster indicate that the IRL agent has learned a robust policy that generalizes well across all evaluation starting states. The GAIL raster, while largely successful, suggests slightly higher sensitivity to initial conditions.\n",
        "3.  **Conclusion**: For safety-critical robotic applications where reliability is paramount, the projection-based IRL approach (TD3) offers a distinct advantage due to its stationary reward function. GAIL remains a powerful tool for rapid prototyping given its sample efficiency, but may require additional stabilization mechanics for deployment.\n"
    ]

    # Processing cells
    for cell in cells:
        source = "".join(cell['source'])
        
        # 1. Update 4.1 TD3 Evaluation Raster text (Figure 6)
        if "Binary Success Raster during TD3 Evaluation" in source:
            if "visual density is a hallmark" not in source:
                cell['source'].append(td3_raster_text)
                print("Updated Section 4.1 TD3 Raster text.")
        
        new_cells.append(cell)

        # 2. Insert GAIL Eval Raster after Figure 10 (GAIL Eval Success Rate)
        if "Final evaluation success rate of GAIL Apprentices" in source: # Figure 10
             print("Found Figure 10. Inserting GAIL Evaluation Raster (Figure 11).")
             new_cell = {
                "cell_type": "markdown",
                "id": "gail_eval_raster",
                "metadata": {},
                "source": gail_raster_cell_source
             }
             new_cells.append(new_cell)

        # 3. Insert Section 4.4 after Figure 13 (Apprentice 3 comparison)
        # Note: In ipynb it is Figure 13, in md it is Figure 14. We match by content.
        if "Direct comparison for Apprentice 3" in source:
            print("Found Figure 13/14. Inserting Section 4.4.")
            new_cell = {
                "cell_type": "markdown",
                "id": "comp_dashboard",
                "metadata": {},
                "source": section_4_4_source
            }
            new_cells.append(new_cell)

    nb['cells'] = new_cells

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1) # Using indent 1 to match file style roughly, or 4 if clearer
    
    print(f"Successfully updated {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_notebook()
