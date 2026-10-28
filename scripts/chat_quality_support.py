"""Keep tool execution intact; rewrite ordinary no-call replies without schemas."""
import copy
import time
from best_of_n import generate_candidate

STANDARD_SYSTEM='你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。'
CONCISE_SYSTEM='你是MiniMind，由Yuyang LU基于jingyaogong/minimind定制的AI助手。请用用户使用的语言，直接、简短、清楚地回答。按用户的要求作答，不复述问题，不添加无关内容；不确定时如实说明。'

def plain_context(messages,system):
    # Preserve prior user/assistant/tool rounds, replace only system instructions.
    return [dict(role='system',content=system)]+copy.deepcopy([m for m in messages if m['role']!='system'])

def tool_activity(candidate):
    return any(x.get('role')=='tool' for x in candidate.get('events',[])) or any('<tool_call' in t for t in candidate.get('turns',[]))

def ordinary_chat(model,tokenizer,messages,*,tools=None,execute_tool=None,seed=42,
                  temperature=.9,max_new_tokens=384,system=CONCISE_SYSTEM,generator=None,
                  chat_model=None,chat_tokenizer=None):
    """The ordinary second pass sees only original context, never the discarded reply.

    Same sampler/seed is used for both phases. Any attempted tool call keeps its
    complete original execution path. Unfinished rewrite falls back to the
    original reply; no partial rewrite enters conversation history.
    """
    gen=generator or generate_candidate
    chat_model=model if chat_model is None else chat_model
    chat_tokenizer=tokenizer if chat_tokenizer is None else chat_tokenizer
    started=time.perf_counter()
    kwargs=dict(execute_tool=execute_tool,seed=seed,temperature=temperature,
        max_new_tokens=max_new_tokens,thinking=False,max_turns=3)
    if not tools:
        chosen=gen(chat_model,chat_tokenizer,plain_context(messages,system),tools=None,**kwargs)
        return dict(chosen,chat_mode='plain',rewritten=False,router=None,rewrite_fallback=False)
    routed=gen(model,tokenizer,copy.deepcopy(messages),tools=tools,**kwargs)
    if tool_activity(routed) or routed.get('failure')=='context_limit':
        return dict(routed,chat_mode='tool_path',rewritten=False,router=None,rewrite_fallback=False)
    rewritten=gen(chat_model,chat_tokenizer,plain_context(messages,system),tools=None,**kwargs)
    use_rewrite=bool(rewritten.get('eligible') and not tool_activity(rewritten))
    chosen=copy.deepcopy(rewritten if use_rewrite else routed)
    chosen.update(chat_mode='rewritten_plain' if use_rewrite else 'rewrite_fallback',rewritten=use_rewrite,
        router=routed,rewrite_attempt=rewritten,rewrite_fallback=not use_rewrite,
        generated_tokens=routed['generated_tokens']+rewritten['generated_tokens'],
        generation_seconds=time.perf_counter()-started)
    return chosen

