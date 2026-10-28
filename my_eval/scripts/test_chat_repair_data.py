"""Scientific-data invariants and no-reference-leak checks before final evaluation."""
import copy
import unittest
from unittest.mock import patch
import chat_repair_common as e
import torch
import eval_chat_repair as evaluator

class DataTests(unittest.TestCase):
    def test_exact_heldout_exclusion_in_all_training_rows(self):
        held={e.old.norm(t['question']) for t in e.read(e.OUT/'dev_tasks.json')+e.read(e.OUT/'final_tasks.json')}
        for f in [e.P/'dataset/dpo_eval300.jsonl',e.P/'my_eval/results/identity_repair_20261024/dev_language100.jsonl']:
            held.update(e.old.norm(e.old.question(r)) for r in e.rows(f))
        for r in e.rows(e.OUT/'train.jsonl'):
            q='\n'.join(m['content'] for m in r['messages'] if m['role']=='user')
            self.assertNotIn(e.old.norm(q),held)
    def test_teacher_quality_decisions_are_applied(self):
        raw=e.read(e.OUT/'teacher_answers.json');cur=e.read(e.OUT/'teacher_curation.json')
        targets={r['messages'][-1]['content'] for r in e.rows(e.OUT/'train.jsonl') if r.get('case_id')}
        self.assertEqual(len(cur['overrides']),43)
        self.assertTrue(all(r['answer'] in targets for k,r in cur['overrides'].items() if k not in cur['excluded']))
        self.assertTrue(all(raw[k]['complete'] for k in raw if raw[k]['accepted']))
        self.assertIn('teacher_197',cur['excluded'])
    def test_every_ordinary_question_has_both_contexts(self):
        groups={}
        for r in e.rows(e.OUT/'train.jsonl'):
            if 'case_id' in r:groups.setdefault(r['case_id'],[]).append(r)
        self.assertEqual(len(groups),760)
        for pair in groups.values():
            self.assertEqual(len(pair),2);self.assertEqual(sum(r['tools'] is None for r in pair),1)
            self.assertEqual(pair[0]['messages'][-1],pair[1]['messages'][-1])
    def test_tool_counts_and_descriptions_jointly_balanced(self):
        d=e.read(e.OUT/'menu_balance_audit.json');values=list(d['cells'].values())
        self.assertEqual(len(values),36);self.assertLessEqual(max(values)-min(values),1)
        self.assertEqual(sum(values),760)
    def test_no_silent_truncation_or_prompt_token_supervision(self):
        audit=e.read(e.OUT/'token_audit.json');data=torch.load(e.OUT/'tokenized.pt',weights_only=True)
        self.assertEqual(audit['rows'],2760);self.assertFalse(audit['excluded']);self.assertEqual(audit['train_sha256'],e.sha(e.OUT/'train.jsonl'))
        # Independent boundary check: everything preceding the first assistant
        # header is context and must carry ignore_index, including tool schemas.
        header=e.tok()('<|im_start|>assistant\n',add_special_tokens=False).input_ids
        for row in data:
            first=next(i for i in range(len(row['ids'])) if row['ids'][i:i+len(header)]==header)
            self.assertTrue(all(v==-100 for v in row['labels'][:first+len(header)]))
            self.assertTrue(any(v!=-100 for v in row['labels']))
    def test_reference_answer_never_reaches_generator(self):
        task=evaluator.chat_tasks('dev')[0];before=copy.deepcopy(task);seen=[]
        def fake(model,tok,messages,**kwargs):
            seen.append((messages,kwargs));return dict(final='probe',turns=['probe'],eligible=True,failure=None,messages=[],events=[],generated_tokens=1,generation_seconds=.1)
        with patch.object(evaluator,'generate_candidate',side_effect=fake),patch.object(evaluator.e,'save'):
            evaluator.generate_rows('unit_reference_probe','not_a_model',None,None,[task])
        self.assertEqual(task,before);self.assertEqual(seen[0][0],task['input_messages'])
        self.assertNotIn('reference_for_review_only',str(seen[0]))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DataTests))
    e.save('tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,errors=len(r.errors),failures=len(r.failures)))
    raise SystemExit(0 if r.wasSuccessful() else 1)
