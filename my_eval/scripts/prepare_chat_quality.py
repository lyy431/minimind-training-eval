"""Write fresh development/final tasks and a content rubric before generation."""
import chat_quality_common as e

# References/rubrics are for review only: never passed to the generator.
DEV={
'knowledge':[
('用一句简单的话解释，为什么用完水龙头要关紧。','节约水、防止滴漏，不编造危害'),
('为什么做事前列个清单会有帮助？用一句话回答。','帮助记住任务或安排顺序'),
('Explain in one sentence what a table of contents helps a reader do.','find sections or topics in a document'),
('为什么把文档起一个有意义的文件名比较好？','便于识别查找整理')],
'writing':[
('替我写一句祝同学考试顺利的话，不要解释。','给出一条祝福，不是讨论如何写'),
('帮我写一句提醒室友轻声说话的话，语气客气。','礼貌请求轻声说话'),
('Write one friendly sentence inviting a friend to read together.','actual friendly invitation to read together'),
('写一句感谢邻居代收快递的话。','直接感谢邻居帮忙收快递')],
'retrieval':[
('通知写着：活动在图书馆二楼举行。活动地点在哪里？','图书馆二楼'),
('便签内容是：陈老师负责钥匙，小赵负责海报。谁负责海报？','小赵'),
('The label says: box A contains pencils; box B contains paper. Which box contains paper?','box B'),
('请原样抄出这段文字中的编号：登记号是Q-572。只输出编号。','Q-572')],
'identity':[
('这个聊天助手叫什么，谁做了本地定制？','MiniMind；Yuyang LU'),
('请用一句话告诉我你的名字及本地定制者。','MiniMind；Yuyang LU'),
('Who customized this MiniMind assistant? Give the full name.','Yuyang LU'),
('原始MiniMind项目和你这个定制版分别由谁负责？','原作jingyaogong/Jingyao Gong；本地定制Yuyang LU')]
}
FINAL={
'knowledge':[
('请用一句白话解释，阅读时做笔记有什么用。','记录重点或帮助理解回忆'),
('为什么要给重要文件留一份副本？一句话即可。','原件丢失损坏时可恢复；不能只同义反复'),
('请简单说明书本封面和目录各有什么用。','封面标识或保护书；目录列章节便查找'),
('为什么在学习中安排短暂休息可能有帮助？','缓解疲劳恢复注意力；不编造医学数据'),
('请解释一下草稿和最终稿的区别。','草稿可修改；最终稿已整理确认'),
('给刚接触电脑的人解释“重命名文件”是什么意思。','改变文件名称，不是删除内容'),
('用一句话解释，把学习任务分成小步骤有什么好处。','更易开始执行或跟踪进度'),
('What is the purpose of a bookmark in a web browser? Answer briefly.','save or return to a web page'),
('Explain the difference between copying a file and moving it.','copy leaves original; move changes location'),
('Why can a calendar help someone remember appointments?','records dates and reminders for appointments'),
('What does saving a document do? Use simple language.','stores current work so it can be opened later'),
('Explain why descriptive folder names can be useful.','help identify contents or find and organize files')],
'writing':[
('请直接写一句祝朋友搬家顺利的话。','实际祝福搬家顺利'),
('帮我写一句向老师请教问题的礼貌开场白。','礼貌请教，不是描述写法'),
('写一句提醒同学按时交作业的话，语气温和。','温和提醒按时交作业'),
('替我写一句感谢朋友借给我笔记的话。','感谢朋友借笔记'),
('请写一句邀请同学一起去图书馆的话。','邀请同学去图书馆'),
('写一句告诉室友我今晚会晚点回来、请不用等我的话。','今晚晚归且不用等两点齐全'),
('请帮我写一句鼓励练习画画的朋友继续坚持的话。','鼓励画画朋友坚持'),
('Write one polite sentence asking a classmate to share their notes.','actual polite request for notes'),
('Write a short apology for replying late to a friend.','apology for late reply'),
('Write one sentence thanking a teammate for checking your draft.','thanks for reviewing draft'),
('Write a friendly invitation to a study group this Saturday.','invitation to study group this Saturday'),
('Write a brief good-luck message for a friend starting a new class.','good luck for new class')],
'retrieval':[
('安排表写着：小孟带胶水，小白带剪刀。谁带剪刀？','小白'),
('借阅记录：林同学周四还书，周同学周一还书。林同学哪天还书？','周四'),
('请只复述这句话中的房间号，不要计算：讨论在房间726进行。','726'),
('记录写着文件名是notes_final.txt。请只输出文件名。','notes_final.txt'),
('请原样抄写括号中的文字，不求值：（903-27）。','903-27；不能算成876'),
('材料：绿色箱子装纸张，黄色箱子装胶带。胶带在哪个箱子？','黄色箱子'),
('请按给定内容回答：取件人是吴女士，寄件人是钱先生。谁是取件人？','吴女士'),
('The note says the meeting moved from Monday to Wednesday. What is the new meeting day?','Wednesday'),
('The card lists: owner Mia, item blue umbrella. Who owns the umbrella?','Mia'),
('Repeat the identifier exactly: ticket ID is R-846. Do not calculate anything.','R-846'),
('Copy the text inside brackets without evaluating it: [728+16].','728+16, not744'),
('A label says shelf C holds maps and shelf D holds notebooks. Where are the maps?','shelf C')],
'identity':[
('请告诉我这款本地定制助手的名称和定制者的全名。','MiniMind；Yuyang LU'),
('如果我要介绍你，应该说你叫什么、由谁定制？','MiniMind；Yuyang LU'),
('你属于哪个模型项目？本地定制版由谁制作？','MiniMind；Yuyang LU'),
('请用简短的话介绍你和Yuyang LU的关系。','MiniMind由Yuyang LU定制；不得声称原作也是他'),
('jingyaogong和Yuyang LU分别与这个助手有什么关系？','原作jingyaogong；本地定制Yuyang LU'),
('请说明MiniMind原项目作者与这个版本定制者的区别。','原作jingyaogong/Jingyao Gong；本地定制Yuyang LU'),
('只回答你的模型名称和本地定制者姓名。','MiniMind；Yuyang LU'),
('What is this assistant called, and who customized this version?','MiniMind; Yuyang LU'),
('Give your model name and the full name of your local customizer.','MiniMind; Yuyang LU'),
('How is Yuyang LU connected to this assistant?','customized MiniMind; not author of original upstream project'),
('Who made the original MiniMind project, and who customized this local assistant?','jingyaogong/Jingyao Gong original; Yuyang LU local'),
('Introduce yourself in one short sentence, including your customizer.','MiniMind; Yuyang LU')]
}

def tasks(source,split):
    result=[]
    for category,pairs in source.items():
        for i,(q,ref) in enumerate(pairs):
            result.append(dict(id=f'{split}_{category}_{i}',category=category,question=q,
                messages=[dict(role='user',content=q)],reference_for_review_only=ref,
                language='en' if q[0].isascii() else 'zh',seed=710000+len(result)*31+(0 if split=='dev' else 5000)))
    return result

def main():
    assert not (e.OUT/'plan.json').exists(),'Prepared tasks are immutable'
    dev=tasks(DEV,'dev');final=tasks(FINAL,'final')
    # Add history to8 predefined final tasks, while keeping the final question's reference unchanged.
    for i in [0,7,12,19,24,31,36,43]:
        final[i]['messages']=[dict(role='user',content='接下来请用简单的话回答。' if final[i]['language']=='zh' else 'Please use simple language for my next question.'),
            dict(role='assistant',content='好的，我会尽量说得简单清楚。' if final[i]['language']=='zh' else 'Sure, I will keep the explanation simple.')]+final[i]['messages']
    keys=[e.old.norm(t['question']) for t in dev+final];assert len(keys)==len(set(keys))
    known=set();sources=[]
    for folder in ['tool_choice_round2_20261026','tool_choice_20261026','identity_repair_20261024']:
        root=e.P/'my_eval/results'/folder
        for name in ['dev_tasks.json','test_tasks.json','train.jsonl','general_train1000.jsonl','identity_train90.jsonl']:
            f=root/name
            if not f.exists():continue
            rows=e.old.rows(f) if f.suffix=='.jsonl' else e.read(f)
            for r in rows:
                q=r.get('question') or e.old.question(r) or '\n'.join(m['content'] for m in r.get('messages',[]) if m['role']=='user')
                if q:known.add(e.old.norm(q))
            sources.append(str(f.relative_to(e.P)))
    overlap=set(keys)&known;assert not overlap,overlap
    e.save('dev_tasks.json',dev);e.save('final_tasks.json',final)
    e.save('rubric.json',dict(created_before_generation=True,scale={'0':'未满足主要请求、基本不可用或有严重可见错误','1':'部分回应但有明显遗漏、混乱、重复或不恰当肯定','2':'主要请求得到清楚实用回应，无可见严重错误'},
        reviewer='assistant; randomized answer identifiers, no reward scores shown; not independent human blind evaluation',
        identity='须区分原始项目作者与本地定制者；名字出现不等于内容正确',
        unusable='empty, unfinished or unexpected tool-call replies cannot receive2',
        references_are_never_model_inputs=True))
    e.save('plan.json',dict(date='2026-10-27',scope='content quality of ordinary chat using frozen v3; no paid calls or new training',
        dev_n=len(dev),final_n=len(final),categories=list(FINAL),final_conditions=['no_tools','six_original','six_reworded'],
        development='v3:standard plain, concise plain,6 original,6 reworded; v2 diagnostic:standard plain,6 original. Select only between standard/concise second-pass prompts on dev.',
        selection='highest equal-condition mean0-2 content score, tie higher score2 fraction, then lower latency; no tools uses single selected prompt, tools retain original route/execution and only regenerate ordinary no-call replies without schemas',
        acceptance='paired final mean0-2 score gain95%CI lower>0 and gain>=.20; usable score2 fraction gain>=10pp; no-tools score2 rate drop<=5pp; historical strict math197 and fresh math48 drops<=5pp; no-tool identity full-content score not worse by>1/14; no actual unselected tool execution',
        statistics='question-cluster averages over3 conditions, paired bootstrap10000 seed20261028; score2 fraction secondary; one seed/question/condition, no optional stopping',
        metadata_in_prompt='concise option includes verified project name and upstream/customizer relation; identity gains there measure prompt-supported behavior, not learned knowledge',
        multi_turn_n=8,overlap_exact_known=0,overlap_sources=sources,semantic_unseen_not_guaranteed=True,
        frozen_inputs=e.metadata(),data_sha256={n:e.sha(e.OUT/n) for n in ['dev_tasks.json','final_tasks.json','rubric.json']}))
    print('Prepared16 development and48 final tasks before generation.',flush=True)

if __name__=='__main__':main()
