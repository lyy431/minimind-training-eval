"""Check routing scores cannot be gamed by empty replies, wrong arguments, or duplicate calls."""
import json
import unittest
import tool_choice_round2 as e

def reply(calls=None,text=None):
    if text is None:text='\n'.join('<tool_call>'+json.dumps(c)+'</tool_call>' for c in calls)
    return dict(turns=[text])
def call(expression,name='calculate_math'):return dict(name=name,arguments=dict(expression=expression))

class Tests(unittest.TestCase):
    def setUp(self):self.task=dict(kind='math',question='计算3+5，另外7*9',expressions=['3+5','7*9'])
    def test_reordered_complete_calls(self):self.assertTrue(e.first_call_metrics(reply([call('7*9'),call('3 + 5')]),self.task)['arguments_ok'])
    def test_missing_and_repeated(self):
        for c in [[call('3+5')],[call('3+5'),call('3+5')]]:self.assertFalse(e.first_call_metrics(reply(c),self.task)['arguments_ok'])
    def test_name_without_correct_arguments(self):
        r=e.first_call_metrics(reply([call('8'),call('7*9')]),self.task)
        self.assertTrue(r['route_ok']);self.assertFalse(r['arguments_ok'])
    def test_wrong_name(self):self.assertFalse(e.first_call_metrics(reply([call('3+5','text_length')]),self.task)['route_ok'])
    def test_nonstring_argument(self):self.assertFalse(e.first_call_metrics(reply([call(8),call('7*9')]),self.task)['arguments_ok'])
    def test_no_tool_empty_and_broken(self):
        for text in ['', '<tool_call', '<tool_call>{bad}</tool_call>', '<tool_call>{}</tool_call>']:
            self.assertFalse(e.first_call_metrics(reply(text=text),dict(kind='no_tool'))['route_ok'])
        self.assertTrue(e.first_call_metrics(reply(text='你好'),dict(kind='no_tool'))['route_ok'])
    def test_menu_keeps_names_schemas(self):
        original={t['function']['name']:t['function']['parameters'] for t in e.menu(8,0,4)}
        for n in [1,2,3,4,5,6]:
            for v in range(7):
                tools=e.menu(n,v,n*100+v);self.assertEqual(len(tools),n)
                self.assertEqual(sum(t['function']['name']=='calculate_math' for t in tools),1)
                for t in tools:
                    params=e.copy.deepcopy(t['function']['parameters']);ref=e.copy.deepcopy(original[t['function']['name']])
                    for x in [params,ref]:
                        for prop in x['properties'].values():prop.pop('description',None)
                    self.assertEqual(params,ref)
    def test_tool_count_independent_of_expression_count(self):
        data=e.rows(e.OUT/'train.jsonl')
        cross=e.Counter((len(e.expressions(r['messages'][0]['content'])),len(r['tools'])) for r in data if r['kind']=='supervised' and r['source_kind']=='math')
        self.assertEqual(len(cross),18);self.assertEqual(set(cross.values()),{80})
    def test_data_disjoint_and_calculator_labels(self):
        train=e.rows(e.OUT/'train.jsonl');dev=e.read(e.OUT/'dev_tasks.json');test=e.read(e.OUT/'test_tasks.json')
        q=lambda rows:{e.norm(m['content']) for r in rows for m in r['messages'] if m['role']=='user'}
        tq=q(train);dq={e.norm(r['question']) for r in dev};fq={e.norm(r['question']) for r in test}
        self.assertFalse(tq&dq or tq&fq or dq&fq)
        for r in dev+test:
            if r['kind']=='math':self.assertEqual([e.value(x) for x in r['expressions']],r['gt'])

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    e.save('tests.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful()))
    raise SystemExit(not r.wasSuccessful())
