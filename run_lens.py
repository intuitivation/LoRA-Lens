import os
import subprocess
import sys

if __name__ == "__main__":
    print("Launching LoRA Lens...")
    # Add project root to PYTHONPATH so 'core' module can be found
    project_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(["streamlit", "run", "web/app.py"], env=env, cwd=project_root)