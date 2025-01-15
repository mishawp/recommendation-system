from pathlib import Path
import os

PYTHONPATH = Path(__file__).resolve().parent.parent
os.chdir(PYTHONPATH)
