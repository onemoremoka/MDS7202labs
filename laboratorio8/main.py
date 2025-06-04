import os
import sys


PATH_F = sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(PATH_F)
if __name__ == "__main__":
    
    os.chdir("apilab")
    os.system("uvicorn apilab.main:app --host 0.0.0.0 --port 8000 --log-level info")