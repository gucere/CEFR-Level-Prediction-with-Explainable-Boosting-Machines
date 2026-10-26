import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "0_Shared_Data_And_Splits"))
from ebm_workflow import main
if __name__ == "__main__":
    main("cefr", None)
