"""Behavior tests: selection, tool isolation, failure recovery, and Streamlit integration."""
import os
from pathlib import Path
P=Path(__file__).resolve().parents[2]
os.environ.update(HF_HOME='D:/conda/hf-cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    PYTHONDONTWRITEBYTECODE='1',TEMP=str(P/'my_eval/runtime'),TMP=str(P/'my_eval/runtime'))
import sys
sys.path[:0]=[str(P),str(P/'scripts')]
import ast
import copy
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch
import torch
import best_of_n as bon
import demo_best_of_support as support


class Batch(dict):
    def __getattr__(self,key):return self[key]
    def to(self,device):return self


class Tok:
    pad_token_id=0
    eos_token_id=2
    def __init__(self,texts):self.texts=texts;self.seen=[]
    def apply_chat_template(self,messages,**kwargs):self.seen.append(copy.deepcopy(messages));return 'prompt'
    def __call__(self,*args,**kwargs):return Batch(input_ids=torch.tensor([[1]]),attention_mask=torch.ones(1,1),token_type_ids=torch.zeros(1,1))
    def decode(self,tokens,**kwargs):return self.texts.pop(0)


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
    def generate(self,**kwargs):
        assert "token_type_ids" not in kwargs
        return torch.tensor([[1,7,2]])


class Tests(unittest.TestCase):
    @contextmanager
    def app_with_result(self, side_effect):
        from streamlit.testing.v1 import AppTest
        from transformers import AutoModelForCausalLM,AutoTokenizer
        prior=os.getcwd();os.chdir(P/'scripts')
        try:
            with patch.object(AutoModelForCausalLM,'from_pretrained',return_value=Model()), \
                 patch.object(AutoTokenizer,'from_pretrained',return_value=Tok([])), \
                 patch.object(support,'SafeRewardSelector',return_value=object()), \
                 patch.object(bon,'best_of_n',side_effect=side_effect) as select:
                app=AppTest.from_file(str(P/'scripts/web_demo.py')).run(timeout=30)
                self.assertEqual(len(app.exception),0)
                option=next(x for x in app.sidebar.checkbox if 'several' in x.label or '多生成' in x.label)
                option.check().run()
                next(x for x in app.sidebar.slider if 'History' in x.label or '历史' in x.label).set_value(2).run()
                yield app,select
        finally:os.chdir(prior)

    def successful_result(self):
        return dict(candidates=[dict(final='selected',events=[dict(role='assistant',content='selected')],
            rm=2,messages=[dict(role='assistant',content='selected')])],selected=0,fallback=False)

    def test_choose_tie_and_prefix(self):
        c=[dict(eligible=True,rm=1),dict(eligible=True,rm=1),dict(eligible=True,rm=3),dict(eligible=True,rm=2)]
        self.assertEqual([bon.choose(c,n) for n in [1,2,4]],[0,0,2])

    def test_exclusion_and_fallback(self):
        c=[dict(eligible=False,rm=99),dict(eligible=True,rm=-3)]
        self.assertEqual(bon.choose(c,2),1)
        c[1]['rm']=float('nan');self.assertEqual(bon.choose(c,2),0)

    def test_no_input_mutation_and_tool_trajectory(self):
        messages=[dict(role='user',content='2+3')];original=copy.deepcopy(messages)
        tok=Tok(['<tool_call>{"name":"calculate_math","arguments":{"expression":"2+3"}}</tool_call>','5'])
        c=bon.generate_candidate(Model(),tok,messages,tools=[dict(function=dict(name='calculate_math'))],
            execute_tool=lambda n,a:dict(result=5))
        self.assertEqual(messages,original);self.assertTrue(c['eligible']);self.assertEqual(c['final'],'5')
        self.assertEqual([x['role'] for x in c['messages']],['assistant','tool','assistant'])
        self.assertEqual(tok.seen[1][-1]['role'],'tool')

    def test_unlisted_tool_never_executed(self):
        tok=Tok(['<tool_call>{"name":"other","arguments":{}}</tool_call>','done'])
        called=[]
        c=bon.generate_candidate(Model(),tok,[dict(role='user',content='hello')],execute_tool=lambda *a:called.append(a))
        self.assertEqual(called,[]);self.assertIn('error',c['messages'][1]['content'])

    def test_unfinished_not_eligible(self):
        c=bon.generate_candidate(Model(),Tok(['<tool_call>']),[dict(role='user',content='hello')])
        self.assertFalse(c['eligible'])

    def test_tool_limit(self):
        c=bon.generate_candidate(Model(),Tok(['<tool_call>{}</tool_call>']),[],max_turns=1)
        self.assertFalse(c['eligible']);self.assertEqual(c['failure'],'tool_turn_limit')

    def test_reward_failure_restores_model(self):
        class Broken:
            def score(self,*args):raise RuntimeError('expected')
        m=Model()
        with self.assertRaisesRegex(RuntimeError,'expected'):
            bon.best_of_n(m,Tok(['a','b']),[],Broken(),n=2)
        self.assertEqual(str(next(m.parameters()).device),'cpu')

    def test_reward_transfer_failure_cleans_partial_gpu_state(self):
        class TransferFailure:
            on_gpu=False
            cpu_calls=0
            def to(self,device):
                self.on_gpu=True
                raise RuntimeError('simulated transfer OOM')
            def cpu(self):
                self.cpu_calls+=1
                self.on_gpu=False
        selector=support.SafeRewardSelector.__new__(support.SafeRewardSelector)
        selector.model=TransferFailure()
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.cuda,'empty_cache') as clear:
            with self.assertRaisesRegex(RuntimeError,'transfer OOM'):
                selector.score([dict(role='user',content='q')],[])
        self.assertFalse(selector.model.on_gpu)
        self.assertEqual(selector.model.cpu_calls,1)
        clear.assert_called_once()

    def test_safe_selector_preserves_successful_scores(self):
        class Fake:
            def cpu(self):pass
        selector=support.SafeRewardSelector.__new__(support.SafeRewardSelector)
        selector.model=Fake()
        candidates=[dict(rm=1.25)]
        with patch.object(bon.RewardSelector,'score',return_value=2.5), \
             patch.object(torch.cuda,'is_available',return_value=False):
            self.assertEqual(selector.score([],candidates),2.5)
        self.assertEqual(candidates,[dict(rm=1.25)])

    def test_trim_context_preserves_entire_tool_round(self):
        history=[dict(role='system',content='old system'),dict(role='user',content='first'),
            dict(role='assistant',content='tool_call'),dict(role='tool',content='result'),
            dict(role='assistant',content='final'),dict(role='user',content='second')]
        before=copy.deepcopy(history)
        result=support.trim_context(history,1,[dict(role='system',content='new system')])
        self.assertEqual([m['role'] for m in result],['system','user','assistant','tool','assistant','user'])
        self.assertEqual(history,before)
        self.assertEqual(support.trim_context(history,0),[dict(role='user',content='second')])
        result[1]['content']='changed'
        self.assertEqual(history,before)

    def test_trim_context_drops_orphan_tools_and_old_rounds(self):
        history=[dict(role='tool',content='orphan'),dict(role='assistant',content='orphan final')]
        for i in range(3):
            history.extend([dict(role='user',content=str(i)),dict(role='assistant',content='answer')])
        history.append(dict(role='user',content='current'))
        result=support.trim_context(history,2)
        self.assertEqual([x['content'] for x in result if x['role']=='user'],['1','2','current'])
        self.assertEqual(result[0]['role'],'user')

    def test_tool_definitions_preserved(self):
        import demo_tools
        old=ast.parse((P/'my_eval/results/best_of_n_20261026/web_demo.before.py').read_text(encoding='utf-8'))
        old_tools=next(ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='TOOLS' for t in n.targets))
        self.assertEqual(old_tools,demo_tools.TOOLS)
        new=ast.parse((P/'scripts/demo_tools.py').read_text(encoding='utf-8'))
        func=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_tool'))
        self.assertEqual(func(old),func(new))

    def test_streamlit_switch_and_selected_history(self):
        from streamlit.testing.v1 import AppTest
        from transformers import AutoModelForCausalLM,AutoTokenizer
        prior=os.getcwd();os.chdir(P/'scripts')
        try:
            with patch.object(AutoModelForCausalLM,'from_pretrained',return_value=Model()), \
                 patch.object(AutoTokenizer,'from_pretrained',return_value=Tok([])), \
                 patch.object(support,'SafeRewardSelector',return_value=object()), \
                 patch.object(bon,'best_of_n',return_value=dict(
                     candidates=[dict(final='first',events=[dict(role='assistant',content='first')],rm=1),
                         dict(final='selected',events=[dict(role='assistant',content='selected')],rm=2,
                              messages=[dict(role='assistant',content='selected')])],selected=1,fallback=False)):
                app=AppTest.from_file(str(P/'scripts/web_demo.py')).run(timeout=30)
                self.assertEqual(len(app.exception),0)
                option=next(x for x in app.sidebar.checkbox if 'several' in x.label or '多生成' in x.label)
                self.assertFalse(option.value)
                option.check().run();self.assertEqual(len(app.exception),0)
                app.chat_input[0].set_value('hello').run()
                self.assertEqual(len(app.exception),0)
                hist=app.session_state['chat_messages']
                self.assertEqual(hist[-1]['content'],'selected')
                self.assertNotIn('first',[x['content'] for x in hist])
        finally:os.chdir(prior)

    def test_ui_failure_rolls_back_and_retry_has_no_failed_prompt(self):
        prior=[dict(role='user',content='previous'),dict(role='assistant',content='old answer')]
        with self.app_with_result([RuntimeError('simulated scoring failure'),self.successful_result()]) as (app,select):
            app.session_state['messages']=copy.deepcopy(prior)
            app.session_state['chat_messages']=copy.deepcopy(prior)
            app.chat_input[0].set_value('failed question').run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.session_state['messages'],prior)
            self.assertEqual(app.session_state['chat_messages'],prior)
            app.chat_input[0].set_value('retry question').run()
            self.assertEqual(len(app.exception),0)
            passed=select.call_args_list[1].args[2]
            self.assertNotIn('failed question',[m['content'] for m in passed])
            self.assertEqual(app.session_state['chat_messages'][-1]['content'],'selected')

    def test_ui_fallback_does_not_commit_dangling_tool_call(self):
        prior=[dict(role='user',content='previous'),dict(role='assistant',content='old answer')]
        dangling='<tool_call>{"name":"calculate_math","arguments":{"expression":"2+3"}}</tool_call>'
        fallback=dict(candidates=[dict(final=dangling,events=[dict(role='assistant',content=dangling)],
            rm=None,messages=[dict(role='assistant',content=dangling)])],selected=0,fallback=True)
        with self.app_with_result([fallback,self.successful_result()]) as (app,select):
            app.session_state['messages']=copy.deepcopy(prior)
            app.session_state['chat_messages']=copy.deepcopy(prior)
            app.chat_input[0].set_value('unfinished question').run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.session_state['chat_messages'],prior)
            self.assertEqual(app.session_state['messages'][-1]['content'],dangling)
            app.chat_input[0].set_value('next question').run()
            self.assertEqual(len(app.exception),0)
            passed=select.call_args_list[1].args[2]
            self.assertNotIn('unfinished question',[m['content'] for m in passed])
            self.assertNotIn(dangling,[m['content'] for m in passed])

    def test_ui_unexpected_exception_rolls_back_then_surfaces(self):
        prior=[dict(role='user',content='previous'),dict(role='assistant',content='old answer')]
        with self.app_with_result([KeyError('unexpected failure')]) as (app,_):
            app.session_state['messages']=copy.deepcopy(prior)
            app.session_state['chat_messages']=copy.deepcopy(prior)
            app.chat_input[0].set_value('failed question').run()
            self.assertEqual(len(app.exception),1)
            self.assertEqual(app.session_state['messages'],prior)
            self.assertEqual(app.session_state['chat_messages'],prior)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (P/'my_eval/results/best_of_n_20261026/tests.json').write_text(json.dumps(
        dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
