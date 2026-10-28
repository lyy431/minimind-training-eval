"""Sequential candidate generation and local reward selection shared by UI/evaluation.

No reference answers are accepted by the selector. Candidate histories are isolated.
"""
import copy
import gc
import math
from pathlib import Path
import random
import re
import json
import threading
import time

import torch
from transformers import AutoModel, AutoTokenizer

INFERENCE_LOCK = threading.RLock()
RM_PATH = Path(__file__).resolve().parents[2] / 'internlm2-1_8b-reward'


def final_answer(text):
    return text.split('</think>')[-1].strip() if '</think>' in text else text.strip()


def choose(candidates, n):
    if not 1 <= n <= len(candidates):
        raise ValueError('Invalid candidate count')
    eligible = [i for i in range(n) if candidates[i].get('eligible', False)
                and isinstance(candidates[i].get('rm'), (int, float))
                and math.isfinite(candidates[i]['rm'])]
    # Preserve the first answer if no complete candidate is available; flag it.
    return max(eligible, key=lambda i: candidates[i]['rm']) if eligible else 0


@torch.inference_mode()
def generate_candidate(model, tokenizer, messages, *, tools=None, execute_tool=None,
                       seed=42, temperature=.9, max_new_tokens=512,
                       thinking=False, max_turns=3):
    started = time.perf_counter()
    history = copy.deepcopy(messages)
    start_len = len(history)
    turns, events, tokens = [], [], 0
    random.seed(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if thinking:
        kwargs['open_thinking'] = True
    if tools:
        kwargs['tools'] = tools
    allowed = {t['function']['name'] for t in tools or []}
    eligible = False
    failure = None
    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(history, **kwargs)
        ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        if ids.input_ids.shape[1] > 4096:
            failure = 'context_limit'
            break
        ids = ids.to(device)
        output = model.generate(input_ids=ids.input_ids, attention_mask=ids.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=.85, top_k=50, repetition_penalty=1.,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new = output[0, ids.input_ids.shape[1]:]
        tokens += len(new)
        text = tokenizer.decode(new, skip_special_tokens=True)
        turns.append(text)
        history.append(dict(role='assistant', content=text))
        events.append(dict(role='assistant', content=text))
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
        if not calls:
            failure = 'unfinished_markup' if '<tool_call>' in text or ('<think>' in text and '</think>' not in text) else None
            if len(new) == max_new_tokens and int(new[-1]) != tokenizer.eos_token_id:
                failure = 'token_limit'
            eligible = bool(final_answer(text)) and failure is None
            break
        if turn == max_turns - 1:
            failure = 'tool_turn_limit'
            break
        for raw in calls:
            try:
                call = json.loads(raw)
                if call.get('name') not in allowed or not isinstance(call.get('arguments', {}), dict):
                    raise ValueError('Tool not enabled or invalid arguments')
                result = execute_tool(call['name'], call.get('arguments', {}))
                name = call['name']
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                result, name = {'error': str(exc)}, 'invalid_call'
            content = json.dumps(result, ensure_ascii=False)
            history.append(dict(role='tool', content=content))
            events.append(dict(role='tool', name=name, content=content))
    return dict(seed=seed, turns=turns, final=final_answer(turns[-1]) if turns else '',
                eligible=eligible, failure=failure, messages=history[start_len:], events=events,
                generated_tokens=tokens, generation_seconds=time.perf_counter()-started)


class RewardSelector:
    def __init__(self, path=RM_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), torch_dtype=torch.float16,
            trust_remote_code=True, local_files_only=True).eval()

    @torch.inference_mode()
    def score(self, messages, candidates, device='cuda'):
        # Match the project's reward context convention. Do not expose tool results
        # or ground-truth answers to the selector; judge the final user-facing text.
        history = '\n'.join(f"{m['role']}: {m['content']}" for m in messages[:-1])
        query = messages[-1]['content'] if messages else ''
        context = f'{history}\n以上是对话历史。我的新问题是：\n{query}' if history else query
        started = time.perf_counter()
        self.model.to(device)
        try:
            for c in candidates:
                c['rm'] = None
                if not c['eligible']:
                    continue
                conv = [dict(role='user', content=context), dict(role='assistant', content=c['final'])]
                rendered = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                if len(self.tokenizer.encode(rendered, add_special_tokens=False)) > 2047:
                    c['eligible'], c['failure'] = False, 'reward_context_limit'
                    continue
                value = float(self.model.get_score(self.tokenizer, conv))
                if not math.isfinite(value):
                    raise ValueError('Non-finite reward score')
                c['rm'] = value  # raw score; clipping would create avoidable ties
        finally:
            self.model.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return time.perf_counter() - started


def best_of_n(model, tokenizer, messages, selector, *, n=4, seed=42, progress=None, **kwargs):
    if n not in (2, 4):
        raise ValueError('Choose 2 or 4 candidates')
    with INFERENCE_LOCK:
        device = next(model.parameters()).device
        candidates = []
        for k in range(n):
            if progress:
                progress(k, n)
            candidates.append(generate_candidate(model, tokenizer, messages, seed=seed+k, **kwargs))
        model.cpu()
        torch.cuda.empty_cache()
        try:
            score_seconds = selector.score(messages, candidates, str(device))
        finally:
            model.to(device)
        return dict(candidates=candidates, selected=choose(candidates, n), score_seconds=score_seconds,
                    fallback=not any(c['eligible'] and c.get('rm') is not None for c in candidates))
