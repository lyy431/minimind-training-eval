"""Independently reconstruct each new identity adapter on its recorded exact base."""
import chat_repair_common as e
import torch

def main():
    torch.set_num_threads(4)
    plan=e.read(e.OUT/'identity_refinement_plan.json');base=e.P/f"out/{plan['base']}_768.pth"
    assert e.sha(base)==plan['base_sha256'];original=torch.load(base,map_location='cpu',weights_only=True)
    records=[]
    for epoch in [1,2,3]:
        name=f'chat_repair_id_e{epoch}';merged=torch.load(e.P/f'out/{name}_768.pth',map_location='cpu',weights_only=True)
        adapter=torch.load(e.P/f'out/{name}_lora_768.pth',map_location='cpu',weights_only=True)
        assert original.keys()==merged.keys()
        for key,p in original.items():
            prefix=key[:-7] if key.endswith('.weight') else ''
            expected=p.float()
            if prefix+'.lora.A.weight' in adapter:expected=expected+adapter[prefix+'.lora.B.weight']@adapter[prefix+'.lora.A.weight']
            assert torch.equal(expected.half(),merged[key]),(name,key)
        records.append(dict(name=name,all_tensors_exact=True,adapter_base=plan['base'],base_sha256=plan['base_sha256']))
    e.save('identity_merge_validation.json',dict(passed=True,rows=records));print('All3 identity merges exactly reconstruct from their recorded e2 base.',flush=True)

if __name__=='__main__':main()
