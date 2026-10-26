import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / '0_Shared_Data_And_Splits'))
from ebm_workflow import main

if __name__ == '__main__':
    main('cefr', ['summarize', *sys.argv[1:]])
