#!/usr/bin/env python3
"""
Fix imports in moved analysis scripts.
Scripts are in src/tools/analysis/ but need to import from project root.
"""
from pathlib import Path
import re

# Analysis directory
ANALYSIS_DIR = Path("src/tools/analysis")

# All Python files in analysis directory
py_files = list(ANALYSIS_DIR.glob("*.py"))

print(f"Found {len(py_files)} Python files to check\n")

for file_path in sorted(py_files):
    if file_path.name == "run_analysis.py":
        continue  # This one is already correctly fixed
    
    print(f"Checking {file_path.name}...")
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = file_path.read_text(encoding='latin-1')
    original = content
    
    # Check if file already has proper ROOT path setup
    if "ROOT = Path(__file__).resolve().parents" in content:
        # Fix the parent level if wrong
        content = re.sub(
            r"ROOT = Path\(__file__\)\.resolve\(\)\.parents\[0\]",
            r"ROOT = Path(__file__).resolve().parents[3]",
            content
        )
    else:
        # Add ROOT setup after imports if not present
        lines = content.split('\n')
        last_import = -1
        
        for i, line in enumerate(lines):
            if line.startswith(('import ', 'from ')):
                last_import = i
        
        if last_import >= 0:
            # Insert ROOT setup after last import
            insert_pos = last_import + 1
            root_setup = [
                "",
                "ROOT = Path(__file__).resolve().parents[3]",
                "if str(ROOT) not in sys.path:",
                "    sys.path.insert(0, str(ROOT))",
            ]
            
            # Check if sys is imported, if not add it
            if "import sys" not in content and "from sys" not in content:
                lines.insert(last_import + 1, "import sys")
                insert_pos += 1
            
            for j, line in enumerate(root_setup):
                lines.insert(insert_pos + j, line)
            
            content = '\n'.join(lines)
    
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print(f"  ✓ Fixed imports\n")
    else:
        print(f"  - No changes needed\n")

print("Done! All imports fixed.")
