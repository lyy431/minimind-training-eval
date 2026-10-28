"""Exercise real shared best_of_n with model offload/reload, not mocked UI calls."""
from eval_best_of_n import load, read, save, OUT, TOOLS
from best_of_n import best_of_n
from demo_best_of_support import SafeRewardSelector
from demo_tools import execute_tool
import copy
import time
import torch

model,tok=load()
selector=SafeRewardSelector()
messages=[dict(role='user',content='请计算 13*17，另外 28+46 等于多少？')]
before=copy.deepcopy(messages)
torch.cuda.reset_peak_memory_stats()
started=time.perf_counter()
result=best_of_n(model,tok,messages,selector,n=2,seed=123456,
    tools=[TOOLS[0],TOOLS[5]],execute_tool=execute_tool,max_new_tokens=512)
elapsed=time.perf_counter()-started
assert messages==before
assert result['candidates'][0]['turns']==read(OUT/'smoke_generated.json')['smoke_math'][0]['turns']
assert str(next(model.parameters()).device).startswith('cuda')
assert str(next(selector.model.parameters()).device)=='cpu'
assert result['candidates'][result['selected']]['rm'] is not None
save('real_ui_path_smoke.json',dict(passed=True,seconds=elapsed,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    input_history_unchanged=True,first_candidate_matches_independent_generation=True,
    generation_model_restored_to_gpu=True,reward_model_offloaded_to_cpu=True,result=result))
print(f'Full shared path passed; {elapsed:.2f}s; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
