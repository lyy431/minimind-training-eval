"""Actual Streamlit single-answer path, with exported v5 and selected calculators."""
import copy
import os
from unittest.mock import patch
import strong_teacher_common as e
e.configure_local()
import torch
from streamlit.testing.v1 import AppTest

def main():
    export=e.read(e.OUT/'demo_export.json');assert export['published']
    previous=os.getcwd();os.chdir(e.P/'scripts');torch.cuda.reset_peak_memory_stats();records=[]
    try:
        app=AppTest.from_file(str(e.P/'scripts/web_demo.py')).run(timeout=60)
        assert not app.exception
        assert app.sidebar.selectbox[0].value=='MiniMind-YuyangLU-v5'
        slider=next(x for x in app.sidebar.slider if 'Max Length' in x.label or '最大生成长度' in x.label)
        assert slider.value==384
        assert 'MiniMind-YuyangLU-v4' in app.sidebar.selectbox[0].options
        cases=[('你是谁？',606000),('请计算27*43，另外612/17',606001),('请写一句祝朋友旅途愉快的话。',606002)]
        for i,(question,seed) in enumerate(cases):
            if i==1:
                for name in ['calculate_math','get_current_time','random_number','text_length','unit_converter','get_current_weather']:
                    next(c for c in app.sidebar.checkbox if c.key=='tool_'+name).check().run(timeout=60)
                    assert not app.exception
                assert len(app.session_state['selected_tools'])==6
                assert next(c for c in app.sidebar.checkbox if c.key=='tool_get_exchange_rate').proto.disabled
            with patch('random.randint',return_value=seed):app.chat_input[0].set_value(question).run(timeout=60)
            assert not app.exception
            messages=app.session_state['messages'];assert messages[-2]['content']==question and messages[-1]['role']=='assistant'
            records.append(dict(question=question,seed=seed,display=messages[-1]['content'],context=copy.deepcopy(app.session_state['chat_messages'])))
            expected=e.read(e.OUT/'demo_smoke.json')['rows'][i]
            actual_assistant=[m['content'] for m in records[-1]['context'] if m['role']=='assistant']
            # History defaults to 0; compare each actual model turn, not tool HTML.
            assert actual_assistant[:-1]==expected['turns'][:-1]
            assert actual_assistant[-1].endswith(expected['turns'][-1]),(question,actual_assistant,expected['turns'])
        e.save('demo_ui_validation.json',dict(passed=True,real_exported_weights=True,default_v5_selected=True,default_tokens384=True,
            old_v4_preserved=True,six_tools_selectable=True,seventh_disabled=True,
            actual_ui_assistant_turns_match_shared_engine=True,rows=records,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        print('Real Streamlit v5 path matches shared engine on identity, math and writing.',flush=True)
        for r in records:print(r['question'],'=>',r['display'][:200].replace('\n',' '),flush=True)
    finally:os.chdir(previous)

if __name__=='__main__':main()
