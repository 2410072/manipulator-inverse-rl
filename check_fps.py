import gymnasium as gym
import panda_gym

env = gym.make("PandaReach-v3")
print(f"Env Metadata: {env.metadata}")
# Check if there is control frequency info
try:
    print(f"Sim dt: {env.sim.dt}")
    print(f"Render FPS: {env.metadata.get('render_fps')}")
except:
    pass
    
env.close()
