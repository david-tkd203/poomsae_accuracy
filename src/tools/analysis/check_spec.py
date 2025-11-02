import json
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = json.load(open('config/patterns/8yang_spec.json'))
m = spec['moves'][0]
print(f"Keys en primer movimiento: {list(m.keys())}")
print(f"\nPrimer movimiento:")
for k, v in list(m.items())[:8]:
    print(f"  {k}: {v}")
