
# compare_utils.py - Utility functions for Compare experiments setup and data processing

def filter_apprentice_data(data_list, exclude_name="TD3_Apprentice_0"):
    """
    Filter out a specific apprentice from the data list by name.
    """
    return [d for d in data_list if d.get('name') != exclude_name]

def prepare_final_comparison_data(td3_expert_eval_data, td3_apprentice_eval_data, gail_apprentice_eval_data):
    """
    Combine evaluation data from Expert, TD3 Apprentices, and GAIL Apprentices
    into a format suitable for final comparative plotting.
    
    Returns:
        expert_data_dict: Dictionary for expert data
        all_apprentices_list: List of dictionaries for all apprentices
    """
    expert_data = {
        'name': 'TD3 Expert',
        'scores': td3_expert_eval_data.get('returns', []),
        'successes': td3_expert_eval_data.get('successes', [])
    }

    all_apprentices = []
    
    # TD3 Apprentices (excluding Apprentice 0)
    for r in td3_apprentice_eval_data:
        if r.get('name') == 'TD3_Apprentice_0':
            continue
        all_apprentices.append({'name': r['name'], 'scores': r['scores'], 'successes': r['successes']})
        
    # GAIL Apprentices
    for r in gail_apprentice_eval_data:
        all_apprentices.append({'name': r['name'], 'scores': r['scores'], 'successes': r['successes']})
        
    return expert_data, all_apprentices
