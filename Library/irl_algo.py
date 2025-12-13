# irl_algo.py - Inverse Reinforcement Learning Algorithms

import torch

def solve_projection_method(expert_feature_expectation, feature_expectation, feature_expectation_bar, i):
    """
    Calculate the next reward weight vector using the Projection Method.
    
    Args:
        expert_feature_expectation (torch.Tensor): The expert's feature expectation vector (µ_E).
        feature_expectation (list): List of feature expectations from previous policies (µ^(0)...µ^(i-1)).
        feature_expectation_bar (list): List of intermediate feature expectations (µbar^(0)...µbar^(i-2)).
        i (int): The current iteration number (apprentice index).
    
    Returns:
        tuple: (weight, margin, new_bar_element)
            - weight (torch.Tensor): The new weight vector w^(i).
            - margin (float): The margin t(i) = ||µ_E - µbar^(i-1)||.
            - new_bar_element (torch.Tensor or None): The new element to append to feature_expectation_bar (µbar^(i-1)), or None if i=1.
    """
    
    # Step 2: IRL
    
    # First iteration of step 2 of algorithm (Apprentice 1)
    if i == 1:
        # µ¯^(0) = µ^(0) -> This is handled by returning it as the new bar element
        new_bar = feature_expectation[i - 1]
        
        # w^(1) = µ_E - µ^(0)
        weight = (expert_feature_expectation - feature_expectation[i - 1]).view(-1, 1)
        
        # Margin: ||µ_E - µ¯^(0)|| -> we use the *previous* bar which is just µ^(0) here effectively
        # Note: logic in notebooks usually appends to bar list first.
        # In notebook:
        # feature_expectation_bar.append(feature_expectation[i - 1])
        # weights.append(...)
        # margins.append((expert_feature_expectation - feature_expectation_bar[i - 1]).norm().item())
        
        margin = (expert_feature_expectation - new_bar).norm().item()
        
        return weight, margin, new_bar

    # Iterations 2 and onward of step 2 of algorithm (Apprentice 2+)
    else:
        # µ¯(i−2) -> feature_expectation_bar[i-2] if we consider 0-based index of bar matches iteration index roughly
        # Note: In notebook loop `for i in range(100)`:
        # i=1: bar appends feature[0]. bar has length 1 (index 0).
        # i=2: uses bar[i-2] which is bar[0].
        # So we need the bar list passed in.
        
        A = feature_expectation_bar[i - 2]              # µ¯(i−2)
        B = feature_expectation[i - 1] - A              # (µ(i−1) − µ¯(i−2))
        C = expert_feature_expectation - A              # µ_E − µ¯(i−2)

        numerator = (B.view(-1, 1).t() @ C.view(-1, 1))
        denominator = (B.view(-1, 1).t() @ B.view(-1, 1))
        
        if denominator == 0:
            denominator = torch.tensor(1e-8)
            
        # µ¯(i-1) = A + ...
        new_bar = A + (numerator / denominator) * B
        
        # w(i) = µ_E − μ¯(i−1)
        weight = (expert_feature_expectation - new_bar).view(-1, 1)

        # t(i) = ∥µ_E − µ¯(i−1)∥_2
        margin = (expert_feature_expectation - new_bar).norm().item()
        
        return weight, margin, new_bar
