import os
import subprocess
import sys

project_root = "/home/sharansh/CRIS"

def audit_imports():
    errors = []
    print("Starting import audit...")
    count = 0
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py") and file != "import_audit.py":
                filepath = os.path.join(root, file)
                module_name = os.path.relpath(filepath, project_root).replace(os.path.sep, ".")[:-3]
                
                if "__pycache__" in filepath:
                    continue
                    
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                count += 1
                if result.returncode != 0:
                    errors.append((module_name, result.stderr.strip()))

    print(f"Audited {count} Python modules.")
    if errors:
        print(f"Found {len(errors)} import errors:")
        for mod, err in errors:
            print(f"Module: {mod}\nError:\n{err}\n{'-'*40}")
    else:
        print("All imports resolved successfully!")

if __name__ == "__main__":
    audit_imports()
