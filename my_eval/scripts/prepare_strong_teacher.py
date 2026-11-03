"""Generate synthetic training requests and freeze new dev/final questions first."""
from collections import Counter
import copy
import random
import strong_teacher_common as e

# These are ordinary concepts, not current events or specialist factual questions.
# Held-out concepts/scenarios below are separate from training concepts/scenarios.
EXPLAIN_ZH = [
 ('为什么保存文件时起一个清楚的名字有用？','便于识别和以后查找文件'),
 ('为什么把一项大任务拆成几个小步骤有帮助？','更容易开始、安排顺序并检查进度'),
 ('为什么阅读一段话时要注意前后句？','上下文帮助理解意思，避免断章取义'),
 ('为什么同一个词在不同句子里可能意思不同？','词义受到上下文影响'),
 ('为什么做记录时写上日期有帮助？','便于判断先后和找到某一天的记录'),
 ('为什么说明路线时可以提到明显的建筑？','地标帮助辨认位置与方向'),
 ('为什么把相似的文件放在一个文件夹里有用？','便于分类管理和查找'),
 ('为什么需要区分文件名和文件内容？','文件名是标识，内容是文件保存的信息'),
 ('为什么把草稿与最终版本标清楚有用？','避免拿错尚未完成的版本'),
 ('为什么多人合作时先说清各自负责什么有用？','减少遗漏和重复劳动'),
 ('为什么安排活动时要告诉大家地点和时间？','参与者才能知道何时到哪里'),
 ('为什么说明步骤时用一、二、三编号有帮助？','清楚展示顺序，便于跟随'),
 ('为什么给物品分类前要先确定分类标准？','让分类一致，避免混乱'),
 ('为什么同一组数据画成图后有时更容易比较？','图形能直观呈现差异和趋势'),
 ('为什么发送消息前检查收件人有用？','避免把消息发错人'),
 ('为什么引用别人说的话时要注明说话的人？','让读者知道观点来源，避免误归属'),
 ('为什么描述一个物品时说出颜色和形状有帮助？','具体特征便于识别物品'),
 ('为什么需要把自己的意见与材料中的事实分开？','便于判断哪些是已给信息，哪些是个人看法'),
 ('为什么比较两个方案时最好采用同一标准？','比较更一致，减少不公平的差异'),
 ('为什么需要把会议中的决定记下来？','方便之后查阅并执行约定'),
 ('为什么给别人建议前先了解对方需求有用？','建议更贴合实际问题'),
 ('为什么解释缩写时可以先写出全称？','帮助不熟悉缩写的人理解'),
 ('为什么给文章分段能帮助阅读？','区分不同意思，让结构更清楚'),
 ('为什么填写表格时要看清每列标题？','避免把内容填进错误的栏目'),
 ('为什么选择例子时要和解释的概念相关？','相关例子能帮助理解概念'),
 ('为什么时间表发生变化时要及时告诉参与者？','方便大家调整安排，避免按旧时间行动'),
 ('为什么比较价格时还要看数量是否相同？','数量不同的总价不能直接反映单位价格'),
 ('为什么写通知时把关键内容放在前面有用？','读者能迅速找到重要信息'),
 ('为什么找不到某条信息时可以换一个关键词搜索？','不同关键词可能匹配到不同相关内容'),
 ('为什么把临时笔记整理成清单有帮助？','便于逐项查看和检查遗漏'),
 ('为什么说明规则时给一个例子有帮助？','把抽象规则变成具体情形'),
 ('为什么需要区分任务完成与任务开始？','两者代表不同进度，不能互相代替'),
 ('为什么复述约定的地点和时间有时有帮助？','可以确认双方理解一致'),
 ('为什么记录数量时还要写单位？','单位明确数值代表的量'),
 ('为什么空白表格不能告诉我们某个人的答案？','空白表格只有结构，没有填写的信息'),
 ('为什么看材料时不能把两个人的事情混在一起？','不同人的信息应对应正确的人'),
 ('为什么答复邀请时明确说能否参加有用？','方便邀请者确定人数与安排'),
 ('为什么短消息通常一次说清一件事更好读？','重点明确，减少理解负担'),
 ('为什么相同的编号可能在不同系统中代表不同物品？','编号的意义取决于所属系统'),
 ('为什么抄写文件名时要保留后缀？','后缀是完整文件名的一部分，可能区分文件类型')]
EXPLAIN_EN = [
 ('Why can a short example make an explanation clearer?','A concrete case helps the reader understand an abstract idea.'),
 ('Why is it useful to name the owner of a task?','It makes responsibility clear and reduces confusion.'),
 ('Why should a calendar entry include a location?','People need to know where the event takes place.'),
 ('Why is a blank form different from a completed form?','A blank form has fields but no filled-in answers.'),
 ('Why should a file name keep its extension when copied?','The extension is part of the name and can indicate the type.'),
 ('Why can a short summary help after a long discussion?','It collects the main points and decisions.'),
 ('Why should a comparison use the same unit on both sides?','Different units can make the numbers misleading.'),
 ('Why is a version number useful on a document?','It helps distinguish revisions.'),
 ('Why can a descriptive email subject help the reader?','It tells the reader what the message concerns.'),
 ('Why should a quote identify its speaker?','It makes the source of the words clear.')]

WRITING_ZH = [
 '感谢同学帮我整理讲义','请朋友把借走的画笔下周带来','告诉组员我已整理好讨论记录',
 '邀请朋友周末一起去图书馆','礼貌询问同学是否方便交换座位','向邻居说明周六下午会搬运一个书柜',
 '提醒队员明天带上活动签到卡','为迟回复朋友的消息道歉','祝贺同学完成自己的第一幅画',
 '鼓励正在准备班级发言的同学','请同学帮我把练习册转交老师','感谢朋友提醒我带雨衣',
 '说明我今天无法参加读书小组','询问朋友什么时候方便归还桌游','感谢组员认真修改我们的海报',
 '请同学发来上次讨论的提纲','邀请朋友一起整理闲置书籍','提醒伙伴把桌面收拾整齐',
 '说明我已经把报名表发过去了','为把朋友的姓名写错而道歉','请同学检查我发来的名单是否有遗漏',
 '感谢邻居帮忙代收包裹','告诉朋友我们的见面地点改为北门','请朋友给我的海报提一点建议',
 '祝同学的新社团活动顺利','感谢同学愿意听我练习发言','邀请朋友下周一起参观画展',
 '礼貌询问老师能否再解释一次作业要求','请伙伴先保留这份文件的旧版本','提醒同学保存自己完成的练习',
 '告诉队员活动提前半小时开始','为临时取消和朋友的散步约定道歉','感谢同学借给我一本故事书',
 '邀请邻居参加楼里的旧物交换活动','请朋友在来访前发条消息','告诉同学我把活动照片整理到文件夹了',
 '提醒大家离开教室前检查自己的东西','祝朋友新的学习计划坚持顺利','询问同学能否把海报字体放大',
 '感谢朋友帮我测试演示文稿','说明我需要再花一天整理材料','请伙伴将讨论结论写在共享文档里',
 '邀请朋友一起挑选礼物','提醒组员用统一的文件命名方式','为误拿同学的笔记本道歉',
 '请同学告诉我集合的准确地点','感谢朋友帮我找到了落下的手套','告诉伙伴我愿意负责活动签到']
WRITING_EN = [
 'thank a classmate for organizing the handouts','ask a friend to bring back a borrowed sketchbook next week',
 'tell your team that the discussion notes are ready','invite a friend to visit the library this weekend',
 'politely ask a classmate to exchange seats','remind a teammate to save the completed document',
 'apologize for replying late to a message','congratulate a friend on finishing their first drawing',
 'ask a teammate to check a list for missing names','thank a neighbor for receiving a parcel',
 'tell a friend that the meeting point is now the north entrance','ask for one suggestion about your poster']

DEV_EXPLAIN = [
 ('为什么整理书架时给不同区域命名有用？','区域名有助于分类和定位书籍'),
 ('为什么活动报名结束后要核对名单？','可以发现重复或遗漏，确认参加者'),
 ('为什么发出修改后的安排时要说明哪些地方变了？','帮助读者找到变化，避免继续采用旧安排'),
 ('为什么一份说明的标题应与内容相符？','标题帮助读者预期和定位内容，避免误导'),
 ('为什么汇总意见时要保留少数人的不同看法？','避免遗漏观点或错误表示所有人都同意'),
 ('为什么写地址时只给街道名有时还不够？','可能仍需门牌等信息定位具体地点'),
 ('为什么解释任务时最好说明完成的标准？','便于执行者判断做到什么程度算完成'),
 ('为什么把东西按大小排好有时便于寻找？','有规律的顺序便于定位目标'),
 ('为什么修改共享文件前了解大家的约定有用？','避免破坏共同结构或覆盖他人的工作'),
 ('Why should a handover note mention unfinished tasks?','It tells the next person what still needs doing.'),
 ('Why is a labeled diagram easier to discuss?','Labels identify the parts people refer to.'),
 ('Why can repeating an unclear instruction help?','It can reveal misunderstandings and allow clarification.')]
FINAL_EXPLAIN = [
 ('为什么报名表上要区分联系人和参加者？','两者可能是不同的人，区分角色避免联系错误'),
 ('为什么摘要不能只保留原文里最有趣的一句话？','摘要应覆盖主要内容，一句话可能遗漏重点'),
 ('为什么比较两张照片的大小时要说明是在比尺寸还是文件容量？','两者是不同量，比较标准需明确'),
 ('为什么安排轮流发言可以帮助小组讨论？','给每个人表达的机会，减少相互打断'),
 ('为什么在流程图中画箭头有用？','箭头表示步骤方向或关系'),
 ('为什么只说我把它放在那里可能让别人不明白？','物品和位置没有明确指代'),
 ('为什么把说明中的必要步骤和可选步骤区分开有用？','读者能知道必须做什么以及哪些可自行选择'),
 ('为什么一个例外不能直接代表所有情况？','单个特殊情况不足以概括整体'),
 ('为什么问卷中两个不同的问题最好分开问？','便于分别理解和回答，避免混淆'),
 ('为什么转告消息时应区分原话与自己的补充？','避免把个人补充误当成原发送者的意思'),
 ('为什么整理清单时重复的项目值得检查？','可能是重复记录，也可能确实需要多份，需要确认'),
 ('为什么材料写了计划周五完成不能当成已经完成？','计划是未来安排，不能证明实际完成'),
 ('为什么交接物品时记录交给了谁有用？','便于确认接收者和后续查找'),
 ('为什么短句中也需要合适的标点？','标点有助于划分意思，减少歧义'),
 ('为什么同名文件放在不同文件夹里仍可能不是同一份文件？','完整位置不同，内容也可能不同'),
 ('为什么读到新的补充说明后有时需要调整原来的理解？','补充信息可能澄清或改变原来的判断'),
 ('为什么报告中列出尚未确认的信息有用？','明确不确定之处，避免读者把猜测当成事实'),
 ('为什么约定集合时说清是哪一个入口有用？','多个入口可能造成走错地点'),
 ('Why should a schedule distinguish a deadline from a start time?','They describe different points in the work.'),
 ('Why might two people read the same short message differently?','Their context or interpretation of ambiguous words can differ.'),
 ('Why should a summary preserve who did what?','Mixing up people changes the meaning of the source.'),
 ('Why is a checklist item more useful when it names a specific action?','The reader can tell what to do and whether it is complete.'),
 ('Why should a draft be marked before it is shared?','Others need to know it may still change.'),
 ('Why can asking for the missing detail be better than guessing?','It avoids inventing information and can produce a correct answer.')]
DEV_WRITING = ['请朋友替我保留一张空白便签','感谢同学帮我检查目录页','告诉伙伴展板已经送到大厅',
 '向同学询问打印材料的截止日期','提醒队员把名牌戴好','为弄皱朋友的纸张道歉',
 '邀请朋友一起试做纸飞机','告诉同学我想换一个讨论主题','感谢同学分享自己的学习记录',
 'ask a friend to save a blank note card for you','thank a classmate for checking the contents page','tell the team the display board has arrived']
FINAL_WRITING = ['感谢同学帮我核对活动名单中的姓名','请伙伴把最终版文件放进指定文件夹',
 '告诉朋友我会在西门的公告栏旁等他','为忘记归还同学的尺子道歉','邀请朋友下周参加手工制作活动',
 '提醒队员提交材料时附上自己的名字','感谢邻居告诉我包裹放在哪里','请同学说明这次讨论需要准备什么',
 '告诉组员我已经把重复的记录标记出来','礼貌拒绝朋友借走我正在使用的相机','鼓励第一次主持读书会的朋友',
 '请伙伴在修改共享表格后通知大家','祝同学的摄影展示顺利','告诉朋友我可能晚到十分钟',
 '感谢同学耐心陪我练习介绍作品','请朋友帮我确认通知上的地点是否正确','提醒同学使用新发的活动时间表',
 '告诉队员我可以负责收集大家的建议','thank a friend for checking the spelling on your invitation',
 'ask a teammate to place the final file in the shared folder','tell a friend you will wait beside the noticeboard',
 'apologize for forgetting to return a borrowed ruler','invite a friend to a craft activity next week',
 'remind participants to put their names on their submissions']

def item(split, category, index, question, reference, language, **extra):
    return dict(id=f'{split}_{category}_{index:04d}', category=category, question=question,
        messages=[dict(role='user', content=question)], reference_for_review_only=reference,
        language=language, seed=830000 + {'train':0,'dev':10000,'final':20000}[split] + index * 17,
        **extra)

def retrieval(split, n):
    rng = random.Random(842000 + {'train':0,'dev':1,'final':2}[split])
    zh = ['林珂','陈潇','周宁','孟夏','乔羽','贺川'] if split=='train' else (['苏岚','温庭','季澄','方沐'] if split=='dev' else ['陆晴','许舟','唐悦','沈禾'])
    en = ['Nora','Evan','Mila','Hugo','Elsa','Oscar'] if split=='train' else (['Cleo','Amir','Esme','Otis'] if split=='dev' else ['Vera','Iris','Theo','Felix'])
    result=[]
    while len(result)<n:
        i=len(result)
        english = ((i//40)%5==4) if split=='train' else i%4==3
        lang='en' if english else 'zh'; a,b=rng.sample(en if english else zh,2)
        # Every format independently cycles both target positions, not tied to entity order.
        k=i%10; target=(i//10)%2; order=(i//20)%2
        v1,v2=[str(v) for v in rng.sample(range(201,999),2)]
        day1,day2=rng.sample(['Monday','Tuesday','Thursday','Friday'] if english else ['周一','周二','周四','周五'],2)
        tag1=f'{split[0].upper()}K-{rng.randrange(1000,9999)}';tag2=f'{split[0].upper()}R-{rng.randrange(1000,9999)}'
        color1,color2=rng.sample(['blue','green','orange','purple'] if english else ['蓝色','绿色','橙色','紫色'],2)
        person=[a,b][target]; val=[v1,v2][target]; day=[day1,day2][target]; tag=[tag1,tag2][target]; color=[color1,color2][target]
        if k==0:
            facts=[f'{a}的柜号是{v1}',f'{b}的柜号是{v2}'] if not english else [f'{a} uses locker {v1}',f'{b} uses locker {v2}']
            ask=f'{person}的柜号是什么？只给出柜号。' if not english else f'What is {person}\'s locker number? Return only the number.'; ans=val
        elif k==1:
            facts=[f'{a}计划{day1}交材料',f'{b}计划{day2}交材料'] if not english else [f'{a} submits the notes on {day1}',f'{b} submits the notes on {day2}']
            ask=f'{person}哪天交材料？只回答日期。' if not english else f'On which day does {person} submit the notes? Return only the day.';ans=day
        elif k==2:
            facts=[f'{a}的登记号是{tag1}',f'{b}的登记号是{tag2}'] if not english else [f'{a}: registration {tag1}',f'{b}: registration {tag2}']
            ask=f'请只抄出{person}的登记号。' if not english else f'Return only {person}\'s registration code.';ans=tag
        elif k==3:
            facts=[f'{a}带着{color1}文件袋',f'{b}带着{color2}文件袋'] if not english else [f'{a} carries a {color1} folder',f'{b} carries a {color2} folder']
            ask=f'{person}的文件袋是什么颜色？只回答颜色。' if not english else f'What color is {person}\'s folder? Return only the color.';ans=color
        elif k==4:
            facts=[f'{a}把书交给了{b}',f'书的登记号为{tag1}'] if not english else [f'{a} handed the book to {b}',f'the book code is {tag1}']
            ask=('谁收到书？' if target else '谁交出了书？')+'只回答姓名。' if not english else ('Who received the book?' if target else 'Who handed over the book?')+' Return only the name.';ans=[a,b][target]
        elif k==5:
            facts=[f'{a}的文件是note_{v1}.txt',f'{b}的文件是list_{v2}.csv'] if not english else [f'{a}: note_{v1}.txt',f'{b}: list_{v2}.csv']
            ask=f'只输出{person}的完整文件名。' if not english else f'Return only the complete file name for {person}.';ans=[f'note_{v1}.txt',f'list_{v2}.csv'][target]
        elif k==6:
            facts=[f'原集合房间是{v1}',f'最新通知改为{v2}'] if not english else [f'the original room was {v1}',f'the latest notice changes it to {v2}']
            ask=('最新房间号是什么？' if target else '原房间号是什么？')+'只回答号码。' if not english else ('What is the latest room?' if target else 'What was the original room?')+' Return only the number.';ans=[v1,v2][target]
        elif k==7:
            facts=[f'{a}的借书登记号为{tag1}',f'没有登记归还日期'] if not english else [f'{a} has loan reference {tag1}',f'no return date is recorded']
            ask='归还日期是什么？如果未给出，只回答“未提供”。' if not english else 'What is the return date? If absent, reply only "Not provided".';ans='Not provided' if english else '未提供'
        elif k==8:
            facts=[f'标签甲：{v1}+{v2}',f'标签乙：{v2}-{v1}'] if not english else [f'label A: {v1}+{v2}',f'label B: {v2}-{v1}']
            ask=f'只原样抄出标签{["甲","乙"][target]}上的算式，不计算。' if not english else f'Copy only the expression on label {["A","B"][target]}, without calculating it.';ans=[f'{v1}+{v2}',f'{v2}-{v1}'][target]
        else:
            facts=[f'名单顺序：{a}、{b}',f'第一人的编号是{tag1}，第二人的编号是{tag2}'] if not english else [f'list order: {a}, {b}',f'first code: {tag1}; second code: {tag2}']
            ask=f'只回答名单中第{target+1}个人的名字。' if not english else f'Return only the name at position {target+1} in the list.';ans=person
        if order:facts.reverse()
        prefix={'train':('记录：','Record: '),'dev':('请读这份便条：','Read this note: '),'final':('根据下面这份交接信息作答：','Use the following handover information: ')}[split][english]
        q=prefix+('；'.join(facts)+'。' if not english else '. '.join(facts)+'. ')+ask
        if any(r['question']==q for r in result):continue
        result.append(item(split,'retrieval',i,q,ans,lang,exact_answer=ans,format_id=k,target_position=target,fact_order=order))
    return result

def instruction(split,n):
    rng=random.Random(843000+{'train':0,'dev':1,'final':2}[split]);result=[]
    for i in range(n):
        english=i%4==3;lang='en' if english else 'zh';k=i%5
        a=f'{split.upper()}-{rng.randrange(10000,99999)}';b=f'BOX-{rng.randrange(10000,99999)}'
        if k==0:
            q=f'把下面文本原样写一遍，只输出文本本身：{a}' if not english else f'Copy exactly this text, with no added words: {a}';ans=a
        elif k==1:
            q=f'请只输出第二个编号：{a}，{b}。不要加说明。' if not english else f'Return only the second code: {a}, {b}. No explanation.';ans=b
        elif k==2:
            q=f'材料只写了编号{a}，没有写负责人。负责人是谁？未给出时只回答“未提供”。' if not english else f'The note contains code {a}, but no owner. Who is the owner? Reply only "Not provided" if missing.';ans='Not provided' if english else '未提供'
        elif k==3:
            q=f'请把“{a}”放在圆括号里，只输出加好括号的文本。' if not english else f'Put "{a}" inside round brackets. Return only the result.';ans=f'({a})'
        else:
            q=f'请把以下两项用一个斜杠连接，不加空格或说明：{a} 和 {b}。' if not english else f'Join {a} and {b} using one slash, without spaces or explanation.';ans=f'{a}/{b}'
        result.append(item(split,'instruction',i,q,ans,lang,exact_answer=ans,format_id=k))
    return result

def prepare():
    assert not (e.OUT/'plan.json').exists(), 'Preparation is immutable; use a new experiment for changes.'
    assert e.sha(e.P/f'out/{e.BASE}_768.pth')==e.BASE_SHA
    train=retrieval('train',400)+instruction('train',100)
    explain=EXPLAIN_ZH+EXPLAIN_EN
    for i,(q,ref) in enumerate(explain):
        en=i>=len(EXPLAIN_ZH)
        prefixes=['请用一两句解释：','请用简单的话回答：','给初学者说明：','请简短说明：'] if not en else ['Explain in one or two sentences: ','Use simple words: ','Explain to a beginner: ','Please answer briefly: ']
        for j,prefix in enumerate(prefixes):train.append(item('train','explanation',i*4+j,prefix+q,ref,'en' if en else 'zh',concept_id=f'train_explain_{i}'))
    for i,scenario in enumerate(WRITING_ZH+WRITING_EN):
        en=i>=len(WRITING_ZH)
        templates=['请帮我写一句话，用于{q}。直接给出消息。','我要{q}，请直接写一条简短消息。','请用自然礼貌的一句话{q}。','帮我写一句用于{q}的消息，不加解释。','情景：{q}。请给出一条可以直接发送的短消息。'] if not en else ['Write one short message to {q}. Give only the message.','I want to {q}. Write the message itself.','In one polite sentence, {q}.','Write a concise message to {q}, without explanation.','Situation: {q}. Give a short message ready to send.']
        for j,t in enumerate(templates):train.append(item('train','writing',i*5+j,t.format(q=scenario),scenario,'en' if en else 'zh',scenario_id=f'train_write_{i}'))
    assert len(train)==1000
    datasets={'train_items.json':train}
    for split,n,explanations,writings in [('dev',12,DEV_EXPLAIN,DEV_WRITING),('final',24,FINAL_EXPLAIN,FINAL_WRITING)]:
        tasks=retrieval(split,n)+instruction(split,n)
        for i,(q,ref) in enumerate(explanations):
            en=i>=n*3//4; tasks.append(item(split,'explanation',i,q+(' Use one or two simple sentences.' if en else '请用一两句直白的话回答。'),ref,'en' if en else 'zh'))
        for i,scenario in enumerate(writings):
            en=i>=n*3//4;q=f'Write a short message to {scenario}. Give the message itself.' if en else f'请直接写一条简短消息，帮我{scenario}。'
            tasks.append(item(split,'writing',i,q,scenario,'en' if en else 'zh'))
        assert len(tasks)==n*4
        datasets[f'{split}_tasks.json']=tasks
    blocked=set()
    prior=e.P/'my_eval/results/chat_repair_20261027'
    for file in ['dev_tasks.json','final_tasks.json','fixed_supervision.json','teacher_requests.json']:
        blocked.update(e.norm(r['question']) for r in e.read(prior/file))
    for file in [e.P/'dataset/dpo_eval300.jsonl',e.P/'my_eval/results/identity_repair_20261024/dev_language100.jsonl']:
        for r in e.rows(file):blocked.update(e.norm(m['content']) for m in r.get('chosen',r.get('conversations',[])) if m['role']=='user')
    sets={k:{e.norm(t['question']) for t in v} for k,v in datasets.items()}
    for k,v in datasets.items():
        assert len(sets[k])==len(v),(k,'duplicate questions')
        assert not sets[k]&blocked,(k,'overlap with known earlier questions')
    assert not sets['train_items.json']&sets['dev_tasks.json'] and not sets['train_items.json']&sets['final_tasks.json'] and not sets['dev_tasks.json']&sets['final_tasks.json']
    for k,v in datasets.items():e.save(k,v,immutable=True)
    public=[{k:r[k] for k in ['id','question','messages','language','category']} for r in train]
    random.Random(2026102707).shuffle(public)
    e.save('teacher_requests.json',public,immutable=True)
    e.save('retention_math640.json',e.read(prior/'math_supervision640.json'),immutable=True)
    e.save('retention_kl600.json',e.read(prior/'retention600.json'),immutable=True)
    identity=[r for r in e.rows(prior/'identity_refinement_train.jsonl') if r['source_kind']=='identity']
    assert len(identity)==240
    e.save('retention_identity240.json',identity,immutable=True)
    # Teacher input contains training prompts only. Gold answers and all evaluation
    # questions are local, never included in the cloud generation upload.
    audit=dict(counts={k:dict(Counter(r['category'] for r in v)) for k,v in datasets.items()},
        languages={k:dict(Counter(r['language'] for r in v)) for k,v in datasets.items()},
        normalized_exact_overlap=0,teacher_inputs_have_no_answers=True,teacher_inputs_have_no_dev_or_final=True,
        source='Synthetic task-authored material; repeated formats/concepts are explicitly recorded, not 1000 independent topics.',
        scope='Chinese-first bilingual short everyday tasks. Exact question exclusion is not proof of upstream semantic novelty.')
    e.save('preparation_audit.json',audit,immutable=True)
    plan=dict(date='2026-10-27',status='prepared_not_trained',baseline=e.BASE,baseline_sha256=e.BASE_SHA,
        objective='Improve MiniMind v4 itself using curated 7B teacher demonstrations; compare against 1.5B with matched questions and updates.',
        cloud=dict(provider='AutoDL',gpu='RTX 4090, actual VRAM to verify',hourly_cny=2.08,budget_cap_cny=30,
            budget_source='Explicit user answer: 最多30元; rate from user screenshot',connection='awaiting SSH command',paid_work_started=False,
            initial_job_max_seconds=7200,initial_job_rent_ceiling_cny=4.16,automatic_shutdown_required=True),
        teachers=dict(strong=dict(repo=e.STRONG_REPO,revision=e.STRONG_REVISION),weak=dict(path=str(e.P.parent/'Qwen2.5-1.5B-Instruct'),weights_sha256=e.sha(e.P.parent/'Qwen2.5-1.5B-Instruct/model.safetensors'))),
        teacher_generation=dict(system=e.TEACHER_SYSTEM,dtype='bfloat16',batch=2,do_sample=False,max_new_tokens=192,max_input_tokens=768,seed=20261027),
        screening='Preserve raw answers. Exact-output tasks require complete output and exact normalized line endings/outer whitespace only. All open answers manually checked for correctness, requested language, relevance and usability with teacher labels hidden; no answer rewrites. Common accepted IDs only enter primary arms. Report each teacher yield separately.',
        matched_data='Same accepted question IDs, same tools/history and same retention rows in both arms. Answer lengths can differ; report tokens, do not claim equal-token compute. Minimum 400 common cases including >=80 writing and >=40 explanation; otherwise stop and report insufficient data.',
        training=dict(epochs=2,lr=0.000008,kl=3.,batch=4,max_seq=1536,seed=42,num_workers=0,math_first_tokens=64,identity_weight=2.,anchor='exact frozen v4',method='full parameters; no adapter merging'),
        selection='Select epoch separately within each arm on development only. Eligibility: no-call>=90%, dev NLL increase<=.05, dev math drop<=10pp, existing dev identity full-content usable count >= v4. Rank content mean, usable rate, then lower NLL. Freeze both selected arms and v4 before generating any formal answers.',
        final=dict(new_questions=96,conditions=e.CONDITIONS,categories=['retrieval','instruction','explanation','writing'],math_old197=True,math_prior48=True,language296=True,identity14=True,thinking=False,best_of_n=False,max_new_tokens=384,temperature=.9),
        acceptance='7B-trained minus v4: mean content gain>=.15 with question-cluster95%CI lower>0, usable gain>=8pp, no-tools usable drop<=5pp, each math drop<=5pp, NLL increase<=.05, identity usable drop<=1/14, ordinary complete-no-call>=90%. A claim that the larger teacher beats 1.5B additionally requires paired content CI lower>0 for 7B-trained minus matched1.5B-trained. No winner switching after formal evaluation.',
        statistics='3 conditions averaged per base question;10000 paired bootstrap seed20261029, mean +/-1SE, better/worse/tied; report category and language descriptively. Single training seed, subjective assistant review is not independent human blind review.',
        scalability='No automatic expansion to10k/20k. First report pilot result and remaining budget.',
        frozen_inputs={k:e.sha(e.OUT/k) for k in list(datasets)+['teacher_requests.json','retention_math640.json','retention_kl600.json','retention_identity240.json','preparation_audit.json']})
    e.save('plan.json',plan,immutable=True)
    print(e.OUT);print(audit)

if __name__=='__main__':prepare()
