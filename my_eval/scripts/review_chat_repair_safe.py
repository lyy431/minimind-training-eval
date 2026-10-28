"""Restrict identity review discovery to raw model outputs, keeping frozen code intact."""
import argparse
from pathlib import Path
from unittest.mock import patch
import chat_repair_common as e
from review_chat_repair import queue

def main(split):
    real_glob=Path.glob
    selected=e.read(e.OUT/'selection.json')['selected']
    allowed={f'identity_{n}.json' for n in [e.BASE,selected]}
    def raw_only(path,pattern):
        files=real_glob(path,pattern)
        return (p for p in files if p.name in allowed) if pattern=='identity_*.json' else files
    with patch.object(Path,'glob',raw_only):queue(split)
    e.save('review_queue_amendment.json',dict(frozen_generation_and_grader_unchanged=True,
        issue='Original identity_*.json discovery also matched identity_merge_validation.json, whose values are metadata rather than model rows.',
        fix='Postprocessing-only whitelist of the two identity raw result filenames; no weights, questions, generation or scoring rules changed.',
        raw_identity_files=sorted(allowed),source_sha256=e.sha(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['dev','final']);a=p.parse_args();main(a.split)
