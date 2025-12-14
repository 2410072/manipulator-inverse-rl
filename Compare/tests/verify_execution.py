
import unittest
from unittest.mock import patch
import sys
import os
from pathlib import Path
import tempfile
import shutil
import torch

# Add Compare directory to path
current_dir = Path(__file__).resolve().parent
compare_dir = current_dir.parent
sys.path.append(str(compare_dir))

# Import runners (will be patched)
import td3_runner
import gail_runner
import config

class TestExecution(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Define test paths
        self.models_dir = self.test_path / "Models"
        self.results_dir = self.test_path / "Results"
        
        # Patch paths in config (though runners might have already imported them)
        # We need to patch where they are USED in the runners
        
    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.test_dir)

    def test_td3_pipeline(self):
        print("\nTesting TD3 Pipeline...")
        
        # Patch constants in td3_runner
        with patch('td3_runner.N_EPISODES_EXPERT', 2), \
             patch('td3_runner.N_EPISODES_APPRENTICE', 2), \
             patch('td3_runner.N_EPISODES_APPRENTICE_0', 2), \
             patch('td3_runner.EXPLORATION_PERIOD', 1), \
             patch('td3_runner.EXPLORATION_PERIOD_EXPERT', 1), \
             patch('td3_runner.BATCH_SIZE', 64), \
             patch('td3_runner.TD3_MODELS_DIR', self.models_dir / "TD3"), \
             patch('td3_runner.TD3_RESULTS_DIR', self.results_dir / "TD3"), \
             patch('td3_runner.EXPERT_MODEL_PATH', self.models_dir / "Expert"), \
             patch('td3_runner.FEATURE_EXPECTATION_EPISODES', 2), \
             patch('td3_runner.FEATURE_EXPECTATION_EPISODES_APPRENTICE', 2), \
             patch('td3_runner.NUM_APPRENTICES', 2): # Test just 2 apprentices (0 and 1)
             
            # 1. Train Expert
            print("  Training Expert...")
            expert, _ = td3_runner.train_expert(n_episodes=2)
            self.assertTrue((self.models_dir / "Expert" / "actor.pth").exists())
            
            # 2. Evaluate Expert
            print("  Evaluating Expert...")
            td3_runner.evaluate_expert(expert, episodes=2)
            
            # 3. Train Apprentices
            print("  Training Apprentices...")
            td3_runner.train_apprentices(expert=expert, m=2)
            self.assertTrue((self.models_dir / "TD3" / "Apprentices" / "Apprentice_0" / "actor.pth").exists())
            self.assertTrue((self.models_dir / "TD3" / "Apprentices" / "Apprentice_1" / "actor.pth").exists())

    def test_gail_pipeline(self):
        print("\nTesting GAIL Pipeline...")
        
        # We need a trained expert for GAIL trajectory collection if not cached
        # For testing, we can mock the expert loader or ensure expert exists
        # Let's quickly create a dummy expert in the temp dir since GAIL needs it
        
        expert_path = self.models_dir / "Expert"
        expert_path.mkdir(parents=True, exist_ok=True)
        # We assume previous test might not have run or we are isolated
        # Ideally we run TD3 expert first or mock the existence check
        
        # Ensure GAIL uses the test directories
        with patch('gail_runner.N_EPISODES_APPRENTICE', 2), \
             patch('gail_runner.EXPLORATION_PERIOD', 1), \
             patch('gail_runner.BATCH_SIZE', 64), \
             patch('gail_runner.GAIL_MODELS_DIR', self.models_dir / "GAIL"), \
             patch('gail_runner.GAIL_RESULTS_DIR', self.results_dir / "GAIL"), \
             patch('gail_runner.TD3_MODELS_DIR', self.models_dir / "TD3"), \
             patch('gail_runner.EXPERT_MODEL_PATH', self.models_dir / "Expert"), \
             patch('gail_runner.EXPERT_TRAJECTORIES_PATH', self.models_dir / "trajectories.pt"), \
             patch('gail_runner.NUM_APPRENTICES', 2), \
             patch('gail_runner.collect_expert_trajectories') as mock_collect, \
             patch('gail_runner.build_expert_loader') as mock_loader:
            
            # Mock loader to return a dummy generator
            # The loader yeilds batches of (states, actions)
            # Obs shape is approx 18 (panda reach) or similar
            # Actions approx 3
            def dummy_loader_gen():
                while True:
                    yield torch.randn(64, 12), torch.randn(64, 3) # Correct shapes: State 12, Action 3 = 15 total
            
            mock_loader.return_value = dummy_loader_gen()
            
            # Mock collect_expert_trajectories to manually create a dummy file if needed
            # But the runner checks path existence.
            # We patched directory, so it won't exist.
            # checks: if not EXPERT_TRAJECTORIES_PATH.exists(): collect...
            # We let it call collect (which is mocked) so it doesn't crash
            
            # But wait, collect_expert_trajectories is imported from a module.
            # We patched 'gail_runner.collect_expert_trajectories' so the CALL inside gail_runner is mocked.
            # However, `_ensure_expert_trajectories` function checks file existence.
            # If we don't create the file, it might complain later if build_expert_loader tries to read it.
            # But we mocked build_expert_loader too! So we are safe.
            
            # Run Training
            print("  Training GAIL Apprentices...")
            gail_runner.train_apprentices()
            
            # Check models created (Apprentice 1 only, since loop starts at 1)
            self.assertTrue((self.models_dir / "GAIL" / "Apprentice_1" / "actor.pth").exists())
            
            # Run Evaluation
            print("  Evaluating GAIL Apprentices...")
            gail_runner.evaluate_apprentices(episodes=2)


if __name__ == '__main__':
    try:
        test = TestExecution()
        test.setUp()
        test.test_td3_pipeline()
        test.test_gail_pipeline()
        print("\nALL VERIFICATION TESTS PASSED.")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.tearDown()
