"""CPU-only check that the published UI permits6 tools and disables the7th."""
import os
import tool_choice_round2 as e
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from transformers import AutoModelForCausalLM,AutoTokenizer

class Fake:
    def half(self):return self
    def eval(self):return self
    def to(self,*a,**k):return self

def main():
    from demo_tool_guard import execute_selected_tool
    executed=[]
    executor=lambda name,args:executed.append((name,args)) or {'result':5}
    call={'name':'calculate_math','arguments':{'expression':'2+3'}}
    assert 'error' in execute_selected_tool(call,[],executor) and not executed
    allowed=[{'function':{'name':'calculate_math'}}]
    assert execute_selected_tool(call,allowed,executor)=={'result':5} and len(executed)==1
    assert 'error' in execute_selected_tool({'name':'calculate_math','arguments':[]},allowed,executor) and len(executed)==1

    assert e.read(e.OUT/'demo_export.json')['published']
    prior=os.getcwd();os.chdir(e.P/'scripts')
    try:
        with patch.object(AutoModelForCausalLM,'from_pretrained',return_value=Fake()),patch.object(AutoTokenizer,'from_pretrained',return_value=Fake()):
            app=AppTest.from_file(str(e.P/'scripts/web_demo.py')).run(timeout=30)
            assert not app.exception
            names=['calculate_math','get_current_time','random_number','text_length','unit_converter','get_current_weather']
            for name in names:
                next(c for c in app.sidebar.checkbox if c.key=='tool_'+name).check().run()
                assert not app.exception
            assert len(app.session_state['selected_tools'])==6
            assert next(c for c in app.sidebar.checkbox if c.key=='tool_get_exchange_rate').proto.disabled
            next(c for c in app.sidebar.checkbox if c.key=='tool_calculate_math').uncheck().run()
            assert len(app.session_state['selected_tools'])==5
            assert not next(c for c in app.sidebar.checkbox if c.key=='tool_get_exchange_rate').proto.disabled
        e.save('demo_ui_validation.json',dict(passed=True,six_tools_selectable=True,seventh_disabled=True,uncheck_reenables=True,disabled_tools_never_execute=True,valid_selected_tool_executes=True,invalid_arguments_rejected=True))
        print('UI6-tool selection check passed.',flush=True)
    finally:os.chdir(prior)

if __name__=='__main__':main()
