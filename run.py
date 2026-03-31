import subprocess
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "website"))
subprocess.run([sys.executable, "app.py"])
