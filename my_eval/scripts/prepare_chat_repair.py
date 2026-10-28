"""Freeze chat-repair questions and supervision sources before teacher/training runs."""
import copy
import random
import re
import chat_repair_common as e

TOPICS=[
('为什么看不清远处的字时要走近一点？','Explain briefly why a ruler is useful for drawing straight lines.'),
('雨伞通常是用来做什么的？','What is a dictionary used for?'),
('为什么出门前检查是否带钥匙有用？','Why is it useful to label a storage box?'),
('剪刀和胶水的用途有什么不同？','What is the difference between a question and an answer?'),
('为什么公共场所需要垃圾桶？','Why can practicing a skill regularly help?'),
('请简单解释耳机的用途。','What does a lamp help you do?'),
('为什么读说明书后再使用新物品比较方便？','Why is it helpful to ask when instructions are unclear?'),
('请用简单的话解释地图有什么用。','Explain the difference between listening and speaking.'),
('橡皮擦有什么用途？','What is the purpose of a page number?'),
('为什么把常用物品放在固定位置比较方便？','What does a printer do?'),
('自行车铃的用途是什么？','Why is it useful to keep a shopping list?'),
('为什么搬运易碎物品时要轻拿轻放？','Explain what a headline tells a reader.'),
('请简单解释照片和文字说明有什么不同。','What is the difference between an empty box and a full box?'),
('为什么向别人借东西前要先问？','Why can checking your work help you find mistakes?'),
('请简单解释门锁的用途。','What does it mean to sort things by color?'),
('为什么桌面上不需要的杂物可以收起来？','Why is it useful to arrive before an event starts?'),
('为什么遇到不理解的词可以查词典？','What does a reminder help someone do?'),
('请简单说明铅笔和橡皮擦如何配合使用。','Explain why a clear sign can help visitors.'),
('为什么共同使用的物品用完要放回原位？','What is the difference between a fact and a personal preference?'),
('请用简单的话解释标签的作用。','Why is it helpful to explain unfamiliar words?')]

WRITING_ZH=[
('向朋友问好','嗨，最近过得怎么样？'),('请同学帮忙关窗','麻烦你帮我关一下窗户，谢谢！'),
('感谢朋友陪我散步','谢谢你陪我散步，和你一起很开心。'),('礼貌拒绝一次聚会邀请','谢谢你的邀请，但这次我没法参加，祝你们玩得开心！'),
('提醒家人出门带钥匙','出门前记得带好钥匙哦。'),('祝老师假期愉快','祝老师假期愉快，好好放松！'),
('邀请朋友一起喝茶','有空一起喝杯茶、聊聊天吗？'),('为不小心碰掉同学的书道歉','抱歉，我不小心碰掉了你的书，我来帮你捡起来。'),
('祝贺朋友学会骑自行车','恭喜你学会骑自行车，你的坚持有了收获！'),('请邻居帮忙看一下门口','方便的话，能麻烦你帮我留意一下门口吗？'),
('感谢同学帮我找回橡皮擦','谢谢你帮我找回橡皮擦，真是帮了大忙！'),('提醒大家活动结束后带走随身物品','活动结束时请记得带走自己的随身物品。'),
('邀请同学参加班级讨论','欢迎你来参加班级讨论，我们很想听听你的想法。'),('鼓励正在练习唱歌的朋友','坚持练习，你会越来越熟练的，加油！'),
('向朋友说明我需要改约见面的时间','不好意思，我需要调整一下见面时间，我们可以另约一个时间吗？'),('向同学借一块橡皮擦','请问能借一下你的橡皮擦吗？谢谢！'),
('感谢老师耐心回答我的问题','谢谢老师耐心解答，我现在明白多了。'),('给即将参加演出的朋友加油','祝你演出顺利，尽情享受舞台！'),
('请朋友把音乐声音调小一点','不好意思，能请你把音乐声音稍微调小一点吗？'),('祝朋友旅途愉快','祝你旅途愉快，一路顺利！')]
WRITING_EN=[
('greet a friend','Hi, how have you been?'),('ask a classmate to close the window','Could you please close the window? Thank you!'),
('thank a friend for walking with you','Thanks for taking a walk with me; I enjoyed your company.'),('politely decline a party invitation','Thank you for inviting me, but I cannot make it this time; have a wonderful party!'),
('remind a family member to take their keys','Please remember your keys before you leave.'),('wish a teacher a pleasant holiday','Have a relaxing and enjoyable holiday!'),
('invite a friend for tea','Would you like to have some tea and catch up?'),('apologize for knocking a book off a desk','I am sorry I knocked your book down; let me pick it up.'),
('congratulate a friend on learning to ride a bicycle','Congratulations on learning to ride a bicycle; your practice paid off!'),('ask a neighbor to watch the front door','Could you please keep an eye on the front door for a moment?'),
('thank a classmate for finding your eraser','Thanks for finding my eraser; I really appreciate it!'),('remind guests to take their belongings','Please remember to take your belongings when you leave.'),
('invite a classmate to a discussion','Please join our class discussion; we would love to hear your ideas.'),('encourage a friend practicing singing','Keep practicing and enjoy singing; you are making progress!'),
('ask a friend to reschedule a meeting','I am sorry, but could we find another time to meet?'),('ask to borrow an eraser','May I borrow your eraser, please?'),
('thank a teacher for explaining something patiently','Thank you for explaining it so patiently; it makes more sense now.'),('wish a friend luck before a performance','Good luck with your performance and enjoy the stage!'),
('ask someone to lower their music volume','Could you please turn the music down a little?'),('wish a friend a pleasant journey','Have a wonderful trip and a safe journey!')]

def main():
    assert not (e.OUT/'plan.json').exists(),'Frozen preparation must not be replaced'
    for name in ['dev_tasks.json','rubric.json']:
        e.save(name,e.read(e.diagnostic.OUT/name))
    e.save('final_tasks.json',e.read(e.diagnostic.OUT/'final_tasks.json'))
    held=e.read(e.OUT/'dev_tasks.json')+e.read(e.OUT/'final_tasks.json')
    blocked={e.old.norm(t['question']) for t in held}
    for f in [e.P/'dataset/dpo_eval300.jsonl',e.P/'my_eval/results/identity_repair_20261024/dev_language100.jsonl']:
        blocked.update(e.old.norm(e.old.question(r)) for r in e.rows(f))
    rng=random.Random(202610272);requests=[];seen=set(blocked)
    source=e.P/'my_eval/results/identity_repair_20261024/general_train1000.jsonl'
    pool=[]
    for row in e.rows(source):
        conv=row['conversations']
        if len(conv)!=2:continue
        q=conv[0]['content']
        if not 8<=len(q)<=180 or e.old.norm(q) in seen:continue
        if re.search(r'你是谁|你的名字|Who.*you|your name|你.*开发|你.*作者|实时|现在.*天气|当前.*汇率|https?://|```',q,re.I):continue
        pool.append(q)
    rng.shuffle(pool);assert len(pool)>=320,len(pool)
    for q in pool[:320]:
        requests.append(dict(id=f'teacher_{len(requests)}',question=q,source_kind='general'));seen.add(e.old.norm(q))
    for a,b in TOPICS:
        for q in [a,b]:
            for prefix in (['请用简短直白的话回答：','请直接用一两句话作答：','新手提问，请简单说明：','请简要回答下面的问题：'] if not q[0].isascii() else ['Answer briefly: ','Use one or two simple sentences: ','Give a short, direct answer: ','Please answer concisely: ']):
                full=prefix+q;assert e.old.norm(full) not in seen;seen.add(e.old.norm(full))
                requests.append(dict(id=f'teacher_{len(requests)}',question=full,source_kind='knowledge'))
    pairs=[]
    for scenario,answer in WRITING_ZH:
        for template in ['请帮我写一句{q}的话。','用一句话{q}，直接给出成句。','请写一句用于{q}的简短消息。','我要{q}，请直接给我一句合适的话。']:
            pairs.append(dict(question=template.format(q=scenario),answer=answer,source_kind='writing'))
    for scenario,answer in WRITING_EN:
        for template in ['Write one sentence to {q}.','Give a short message to {q}.','Please {q} in one friendly sentence.','I want to {q}. Write the message itself.']:
            pairs.append(dict(question=template.format(q=scenario),answer=answer,source_kind='writing'))
    names=['小罗','小陶','小叶','小卫','小杜','小丁','小梅','小顾','小邵','小魏'];english=['Zara','Owen','Tess','Leo','Ivy','Finn','Ada','Noah','Lena','Ravi']
    colors=['紫色','黑色','橙色','白色','棕色'];en_colors=['purple','black','orange','white','brown']
    retrieval_seen=set()
    for i in range(320):
        k=i%8
        while True:
            a,b=rng.sample(names if k<4 else english,2);color=rng.choice(colors if k<4 else en_colors);room=rng.randint(110,989);tag=f'{rng.choice("HJKLMT")}-{rng.randint(100,999)}'
            if k==0:q=f'记录：{a}负责杯子，{b}负责盘子。谁负责盘子？';ans=b
            elif k==1:q=f'纸上写着：{a}拿了{color}背包。{a}的背包是什么颜色？';ans=color
            elif k==2:q=f'登记信息：房间号{room}，凭证编号{tag}。请只输出凭证编号。';ans=tag
            elif k==3:
                expr=f'{room}+{rng.randint(10,99)}';q=f'请原样抄出这段文本，不要求值：{expr}';ans=expr
            elif k==4:q=f'A record says {a} has the cup and {b} has the plate. Who has the plate?';ans=b
            elif k==5:q=f'The note says {a} chose the {color} backpack. What color is the backpack?';ans=color
            elif k==6:q=f'Entry: room {room}; reference {tag}. Return only the reference.';ans=tag
            else:
                expr=f'{room}-{rng.randint(10,99)}';q=f'Copy this text exactly, without evaluating it: {expr}';ans=expr
            if q not in retrieval_seen:retrieval_seen.add(q);break
        pairs.append(dict(question=q,answer=ans,source_kind='retrieval'))
    idqs=[('这个助手叫什么？','我是MiniMind。'),('请问你是哪个AI助手？','我是MiniMind，由Yuyang LU基于jingyaogong/minimind定制。'),
        ('你的本地定制开发者是谁？','我的本地定制开发者是Yuyang LU。'),('说一下这款助手的名称和本地定制者。','这款助手叫MiniMind，由Yuyang LU进行本地定制。'),
        ('jingyaogong是原作者还是本地定制者？','jingyaogong是原始MiniMind项目的作者，本地定制者是Yuyang LU。'),
        ('原项目是谁写的，这个版本又是谁定制的？','原始MiniMind项目由jingyaogong开发，这个本地版本由Yuyang LU定制。'),
        ('What is this assistant named?','This assistant is named MiniMind.'),('Who is the local customizer of this assistant?','The local customizer is Yuyang LU.'),
        ('Who authored upstream MiniMind and who customized this version?','jingyaogong authored the original MiniMind project, and Yuyang LU customized this local version.'),
        ('Give a short description of your identity.','I am MiniMind, a local assistant customized by Yuyang LU based on jingyaogong/minimind.')]
    for q,ans in idqs:
        for prefix in (['请简短回答：','我想知道：','请告诉我：','简单说一下：','请直接回答：','请说明：','请准确回答：','我有个问题：'] if not q[0].isascii() else ['Please answer: ','I would like to know: ','Briefly: ','Tell me: ','Answer directly: ','Please explain: ','Give a short answer: ','My question is: ']):
            pairs.append(dict(question=prefix+q,answer=ans,source_kind='identity'))
    assert len(pairs)==560,len(pairs)
    for r in pairs:assert e.old.norm(r['question']) not in blocked
    assert len({e.old.norm(r['question']) for r in pairs+requests})==len(pairs+requests)
    e.save('teacher_requests.json',requests);e.save('fixed_supervision.json',pairs)
    oldtrain=e.rows(e.old.OUT/'train.jsonl');math=[r for r in oldtrain if r['kind']=='supervised' and r['source_kind']=='math'];retain=[r for r in oldtrain if r['kind']=='retain']
    rng.shuffle(math);rng.shuffle(retain);assert len(math)>=640 and len(retain)>=600
    e.save('math_supervision640.json',math[:640]);e.save('retention600.json',retain[:600])
    e.save('plan.json',dict(date='2026-10-27',objective='Improve MiniMind weights themselves; user explicitly selected training rather than a runtime teacher backend',
        baseline=e.BASE,baseline_sha256=e.sha(e.P/f'out/{e.BASE}_768.pth'),teacher='D:/minimind/Qwen2.5-1.5B-Instruct',
        local_only=True,no_paid_or_network_calls=True,teacher_requests=len(requests),fixed_supervision=len(pairs),
        ordinary_expansion='each ordinary case once without tools with existing UI system, once with1-6 varied tools/descriptions and no additional system; answers are short complete messages',
        training=dict(epochs=3,lr=.00001,kl=2.,batch=4,max_seq=1536,num_workers=2,seed=42,recipe='full parameter supervised fine tuning from exact v3; CE per row, math first64 answer tokens weight2; KL on600 prior retention rows'),
        dev=dict(chat_questions=16,conditions=3,math_source='round2 development math tasks',language_n=100),
        final=dict(chat_questions=48,conditions=3,history_math=197,other_math=48,identity_checks=14,language_n=296),
        acceptance='chat score0-2 gain>=.20 and paired95%CI lower>0; nonidentity score gain>=.20 and CI lower>0; usable score2 fraction gain>=10pp; no-tools usable drop<=5pp; both math groups drop<=5pp; identity full-content usable count drop<=1/14; NLL296 increase<=.05; ordinary nonempty no-call>=90%',
        selection='Choose epoch on dev only: no-call>=90%, devNLL increase<=.05, devmath strict drop<=10pp; rank mean content score then usable fraction then lower loss; final never used for selection',
        statistics='question cluster means over3 conditions;10000 paired bootstrap resamples seed20261028; single training/generation seed; exploratory development is separate',
        evaluation_sources='carry forward previously prepared16/48 prompts; no formal answer has been generated or inspected',
        data_leakage='normalized exact questions excluded from development, final, NLL100/300; not a claim of complete upstream semantic novelty',
        input_sha256={n:e.sha(e.OUT/n) for n in ['dev_tasks.json','final_tasks.json','rubric.json','teacher_requests.json','fixed_supervision.json','math_supervision640.json','retention600.json']}))
    e.save('development_diagnostic_reference.json',dict(folder=str(e.diagnostic.OUT),formal_generation_started=False,reason='Prompt-only alternatives and swapping small models remained weak on development; retained as exploratory diagnostics, no release. User chose to train MiniMind itself.'))
    print('Prepared480 teacher requests,560 fixed cases,640 math CE,600 retention; final held out.',flush=True)

if __name__=='__main__':main()
