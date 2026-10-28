"""CPU-only check that the published UI permits6 tools and disables the7th."""
import os
import tool_choice_v2 as e
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from transformers import AutoModelForCausalLM,AutoTokenizer

class Fake:
    def half(self):return self
    def eval(self):return self
    def to(self,*a,**k):return self

def main():
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
        e.save('demo_ui_validation.json',dict(passed=True,six_tools_selectable=True,seventh_disabled=True,uncheck_reenables=True))
        print('UI6-tool selection check passed.',flush=True)
    finally:os.chdir(prior)

if __name__=='__main__':main()
