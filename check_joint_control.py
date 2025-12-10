import gymnasium as gym
import panda_gym

try:
    # Attempt to create environment with joint control
    env = gym.make("PandaReach-v3", control_type="joint")

    print(f"--- Environment: {env.spec.id} (Joint Control) ---")
    print(f"Action Space: {env.action_space}")
    print(f"Action Space Shape: {env.action_space.shape}")
    
    obs = env.reset()[0]
    print(f"Observation Space Keys: {list(obs.keys())}")
    
    # Check if joint positions are now in observation
    if 'observation' in obs:
        print(f"Observation Vector Length: {len(obs['observation'])}")
        # Usually obs includes joint position (7) + joint velocity (7) if fully observable?
        # Or maybe it stays same but action changes.
        
except Exception as e:
    print(f"Failed to load with control_type='joint': {e}")
