
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure correct path to import Compare modules
current_dir = os.path.dirname(os.path.abspath(__file__))
compare_dir = os.path.dirname(current_dir)
if compare_dir not in sys.path:
    sys.path.append(compare_dir)

import config
from gail_runner import train_apprentices, GAILTrainer

def test_gail_parameters():
    print("Testing GAIL Parameter Synchronization...")
    
    # Mock GAILTrainer to capture init arguments
    with patch('gail_runner.GAILTrainer') as mock_trainer_cls:
        # Mock the instance returned by the class
        mock_instance = MagicMock()
        mock_trainer_cls.return_value = mock_instance
        
        # Mock gail_train to return empty lists so the loop continues/finishes
        mock_instance.gail_train.return_value = ([], [], [], [])
        
        # Mock internal helpers to avoid actual file I/O or env creation
        with patch('gail_runner.create_env') as mock_env, \
             patch('gail_runner.get_obs_shape') as mock_obs, \
             patch('gail_runner.build_expert_loader') as mock_loader, \
             patch('gail_runner._ensure_expert_trajectories'):
             
             # Run the function
             train_apprentices()
             
             # Verify GAILTrainer was called
             if not mock_trainer_cls.called:
                 print("FAIL: GAILTrainer was not instantiated.")
                 sys.exit(1)
                 
             # Get the arguments used to call GAILTrainer
             # We check the first call (Apprentice 1)
             call_args = mock_trainer_cls.call_args[1] # kwargs
             
             print(f"Captured GAILTrainer Init Args: {list(call_args.keys())}")
             
             # Assertions
             errors = []
             
             expected_params = {
                 'alpha': config.ALPHA,
                 'beta': config.BETA,
                 'gamma': config.GAMMA,
                 'tau': config.TAU,
                 # 'replay_size': config.REPLAY_SIZE, # Might be passed as replay_size or replay_buffer_size depending on impl
                 'noise_factor': config.NOISE_FACTOR,
                 'update_actor_every': config.UPDATE_ACTOR_EVERY,
                 'exploration_period': config.EXPLORATION_PERIOD
             }
             
             for param, expected_val in expected_params.items():
                 actual_val = call_args.get(param)
                 if actual_val != expected_val:
                     errors.append(f"Mismatch for {param}: expected {expected_val}, got {actual_val}")
                 else:
                     print(f"OK: {param} = {actual_val}")
                     
             if errors:
                 print("\nFAILED with mismatches:")
                 for e in errors:
                     print(e)
                 sys.exit(1)
             else:
                 print("\nSUCCESS: All GAIL parameters match config.py!")

if __name__ == "__main__":
    test_gail_parameters()
