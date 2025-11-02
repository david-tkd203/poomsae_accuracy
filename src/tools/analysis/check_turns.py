import json
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = json.load(open('config/patterns/8yang_spec.json'))

# Ver qué valores de "turn" hay
turns = set()
for m in spec['moves']:
    turn = m.get('turn', 'NONE')
    turns.add(turn)

print(f"Valores únicos de 'turn' en la especificación:")
for t in sorted(turns):
    count = sum(1 for m in spec['moves'] if m.get('turn', 'NONE') == t)
    print(f"  {t:20s}: {count:2d} movimientos")

print(f"\nTotal: {len(spec['moves'])} movimientos")

# Ver algunos ejemplos de movimientos con diferentes turn
print(f"\nEjemplos:")
for turn_val in ['NONE', 'RIGHT_90', 'TURN_180']:
    for m in spec['moves']:
        if m.get('turn', 'NONE') == turn_val:
            print(f"  turn={turn_val:15s}: {m['idx']}{m.get('sub', '')} - {m['tech_es'][:40]}")
            break
