import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

files = [
    'data/annotations/moves/8yang_001_moves.json',
    'data/annotations/moves/8yang_001_moves_NEW.json',
    'data/annotations/moves/8yang_001_scored_moves.json',
    'data/annotations/moves/8yang_001_moves_FINAL36.json'
]

for f in files:
    p = Path(f)
    if p.exists():
        data = json.loads(p.read_text())
        count = len(data.get('moves', []))
        print(f"{p.name}: {count} movimientos")
    else:
        print(f"{p.name}: NO EXISTS")
