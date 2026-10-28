"""Behavioral checks for discarded drafts, tool execution and rewrite recovery."""
import unittest
import copy
import chat_quality_common as e
from chat_quality_support import ordinary_chat,plain_context,tool_activity

def answer(text='完成',**extra):
    return dict(final=text,turns=[text],events=[dict(role='assistant',content=text)],
        messages=[dict(role='assistant',content=text)],eligible=True,failure=None,
        generated_tokens=3,generation_seconds=.1,**extra)

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.messages=[dict(role='user',content='写一句祝福')]
        self.tools=[{'function':{'name':'calculate_math'}}]
    def generator(self,outputs):
        calls=[]
        def gen(model,tok,messages,**kwargs):
            calls.append((copy.deepcopy(messages),kwargs));return copy.deepcopy(outputs[len(calls)-1])
        return gen,calls
    def test_plain_is_one_pass(self):
        gen,calls=self.generator([answer()]);ordinary_chat(None,None,self.messages,generator=gen)
        self.assertEqual(len(calls),1);self.assertIsNone(calls[0][1]['tools'])
    def test_draft_never_enters_rewrite_context_or_committed_history(self):
        gen,calls=self.generator([answer('不对题的草稿'),answer('祝你顺利！')]);before=copy.deepcopy(self.messages)
        r=ordinary_chat(None,None,self.messages,tools=self.tools,generator=gen,seed=73)
        self.assertTrue(r['rewritten']);self.assertEqual(r['messages'],answer('祝你顺利！')['messages'])
        self.assertEqual(self.messages,before);self.assertNotIn('不对题的草稿',str(calls[1][0]))
        self.assertIsNone(calls[1][1]['tools']);self.assertEqual(calls[0][1]['seed'],calls[1][1]['seed'])
        self.assertEqual(r['generated_tokens'],6)
    def test_tool_path_is_identical_and_not_reexecuted(self):
        a=answer('5');a['turns']=['<tool_call>{}</tool_call>','5'];a['events'].insert(0,dict(role='tool',name='calculate_math',content='5'))
        gen,calls=self.generator([a]);r=ordinary_chat(None,None,self.messages,tools=self.tools,generator=gen)
        self.assertEqual(len(calls),1)
        for key in a:self.assertEqual(r[key],a[key])
    def test_incomplete_call_keeps_original_path(self):
        gen,calls=self.generator([answer('<tool_call')]);r=ordinary_chat(None,None,self.messages,tools=self.tools,generator=gen)
        self.assertEqual(len(calls),1);self.assertFalse(r['rewritten'])
    def test_ineligible_rewrite_falls_back_without_history_leak(self):
        b=answer('截断');b.update(eligible=False,failure='token_limit')
        gen,calls=self.generator([answer('原回复'),b]);r=ordinary_chat(None,None,self.messages,tools=self.tools,generator=gen)
        self.assertTrue(r['rewrite_fallback']);self.assertEqual(r['messages'],answer('原回复')['messages'])
    def test_tool_marker_in_rewrite_is_rejected(self):
        gen,calls=self.generator([answer('原回复'),answer('<tool_call>{}</tool_call>')])
        r=ordinary_chat(None,None,self.messages,tools=self.tools,generator=gen);self.assertTrue(r['rewrite_fallback'])
    def test_history_retains_previous_complete_tool_round(self):
        hist=[dict(role='system',content='旧说明'),dict(role='user',content='2+3'),dict(role='assistant',content='<tool_call>calc</tool_call>'),dict(role='tool',content='5'),dict(role='assistant',content='5'),*self.messages]
        clean=plain_context(hist,'新说明');self.assertEqual(clean[1:],hist[1:]);self.assertEqual(clean[0]['content'],'新说明')
    def test_context_limit_does_not_retry(self):
        a=answer();a.update(eligible=False,failure='context_limit');gen,calls=self.generator([a])
        ordinary_chat(None,None,self.messages,tools=self.tools,generator=gen);self.assertEqual(len(calls),1)
    def test_separate_chat_backend_keeps_router_on_original_model(self):
        used=[]
        def gen(model,tok,messages,**kwargs):used.append((model,tok));return answer()
        ordinary_chat('router','router_tok',self.messages,tools=self.tools,generator=gen,chat_model='chat',chat_tokenizer='chat_tok')
        self.assertEqual(used,[('router','router_tok'),('chat','chat_tok')])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    e.save('tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)))
    raise SystemExit(0 if r.wasSuccessful() else 1)
