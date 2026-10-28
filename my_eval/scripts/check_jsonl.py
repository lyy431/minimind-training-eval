import json
import sys

for path in sys.argv[1:]:
    n, bad, first = 0, 0, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if first is None:
                    first = obj
            except json.JSONDecodeError:
                bad += 1
            n += 1
    print(f"{path}\n  samples: {n:,}  bad lines: {bad}")
    print("  first sample:", json.dumps(first, ensure_ascii=False)[:300])
