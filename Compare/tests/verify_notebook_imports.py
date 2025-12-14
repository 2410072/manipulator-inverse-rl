
import sys
import os

# Ensure correct path to import Compare modules
current_dir = os.path.dirname(os.path.abspath(__file__))
compare_dir = os.path.dirname(current_dir)
if compare_dir not in sys.path:
    sys.path.append(compare_dir)

print("Verifying Notebook Imports...")

try:
    import config
    print("OK: import config")
    
    import networks
    print("OK: import networks")
    
    import replay
    print("OK: import replay")
    
    import td3_algo
    print("OK: import td3_algo")
    
    import gail_algo
    print("OK: import gail_algo")
    
    import plotting
    print("OK: import plotting")
    
    import irl_utils
    print("OK: import irl_utils")
    
    import documentation
    print("OK: import documentation")
    
    import td3_runner
    print("OK: import td3_runner")
    
    import gail_runner
    print("OK: import gail_runner")
    
    print("\nSUCCESS: All critical modules imported successfully!")

except Exception as e:
    print(f"\nFAIL: Import error: {e}")
    sys.exit(1)
