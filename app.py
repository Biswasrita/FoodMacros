import sys
import os

# Redirect to llm-architecture/app.py
target_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm-architecture", "app.py")

with open(target_script, "r", encoding="utf-8") as f:
    code = f.read()

# Execute target script in current namespace
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm-architecture"))
exec(compile(code, target_script, 'exec'))
