#!/usr/bin/env python
"""
Sign Language Detector - Single Command Launcher
Just run: python run.py
"""
import subprocess
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run the detector
python_exe = os.path.join("mp_env", "Scripts", "python.exe")
if os.path.exists(python_exe):
    subprocess.run([python_exe, "InferenceDataset.py"])
else:
    print("ERROR: Virtual environment not found!")
    print("Please ensure mp_env exists in the project directory.")
    sys.exit(1)
