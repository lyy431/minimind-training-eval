"""Print digests of the eval split and the 3000-problem training sample, exactly as cloud_train.sh builds them."""
import hashlib
import random
import sys

lines = [l.rstrip("\n") + "\n" for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
held = set(random.Random(123).sample(range(len(lines)), 200))
evals = [l for i, l in enumerate(lines) if i in held]
train = [l for i, l in enumerate(lines) if i not in held]
random.seed(42)
used = random.sample(train, 3000)
digest = lambda xs: hashlib.sha256("".join(xs).encode()).hexdigest()[:16]
print(sys.version.split()[0], "| eval200", digest(evals), "| train3000", digest(used))
