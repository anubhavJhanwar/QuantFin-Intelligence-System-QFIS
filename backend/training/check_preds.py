import json, tempfile
from pathlib import Path

tmp = Path(tempfile.gettempdir())
for key, fname in [("BASE", "qfis_base_preds.json"), ("FT", "qfis_ft_preds.json")]:
    path = tmp / fname
    if not path.exists():
        print(f"{key}: file not found"); continue
    data = json.load(open(path, encoding="utf-8"))[:6]
    print(f"\n{'='*60}")
    print(f"  {key} MODEL PREDICTIONS")
    print(f"{'='*60}")
    for r in data:
        print(f"  Q:    {r['question'][:65]}")
        print(f"  REF:  {r['reference']}")
        print(f"  PRED: {r['prediction']}")
        print()
