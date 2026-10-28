"""Keep raw teacher generations; record scope exclusions and explicit corrections."""
import re
import chat_repair_common as e

def main():
    assert not (e.OUT/'train.jsonl').exists()
    data=e.read(e.OUT/'teacher_answers.json');excluded={};overrides={}
    # Ordinary chat CE must not teach direct arithmetic instead of calculator use.
    math=r'\d|计算|算出|加法|减法|乘法|除法|方程|加仑|盎司|arithmetic|calculate|equation|probability'
    specialist=r'治疗|症状|药物|疾病|癌|外阴|手术|医生|临床|legal|medical|symptom|treatment|cancer|diagnos|法律|刑事|诉讼|合同法|投资|股票|枪|shoot|firearm|weapon|bomb|explosive|杀人|恐怖袭击|自杀|自残|色情'
    experience=r'你看过|你去过|你吃过|你用过|have you (seen|visited|used)|do you like|your favorite|你喜欢'
    for name,r in data.items():
        if not r['accepted']:continue
        if r['source_kind']=='general':
            if re.search(math,r['question'],re.I):excluded[name]='calculation/numerical query outside ordinary CE scope'
            elif re.search(specialist,r['question'],re.I):excluded[name]='specialist/sensitive topic outside this small everyday-chat experiment'
            elif re.search(experience,r['question'],re.I):excluded[name]='personal-experience question; avoid unsupported first-person claims'
        if '剪刀和胶水的用途有什么不同' in r['question']:
            overrides[name]=dict(original=r['answer'],answer='剪刀用来裁剪材料，胶水用来把材料粘在一起。',reason='The sampled teacher response incorrectly restricted glue to hard materials; correct all4 paraphrases.')
    # The old general pool contains underspecified and plainly incorrect labels.
    # Keep only the40 directly inspected, straightforward cases; raw answers stay intact.
    inspected_general={2,19,20,25,30,38,49,50,52,59,61,64,65,66,74,89,93,94,96,97,217,219,225,229,231,237,241,242,243,252,256,262,267,277,278,283,294,303,313,319}
    for name,r in data.items():
        if r['accepted'] and r['source_kind']=='general' and int(name.split('_')[1]) not in inspected_general:
            excluded.setdefault(name,'Not retained after content inspection: old-pool labels are not reliable enough for this everyday-chat study')
    fixes=[(range(320,323),'靠近后，字在视野中显得更大，通常更容易辨认。','Incorrect explanation based on reduced light or rays converging'),
        (range(324,328),'A ruler provides a straight edge to guide your pen or pencil.','Explain the straight edge rather than fixed length'),
        (range(356,360),'Regular practice helps reinforce what you have learned and improve your skill.','Avoid treating all skills as muscle memory'),
        (range(362,364),'耳机把音频信号转换为声音，供使用者听音乐、通话或收听其他内容。','One answer reversed the sound/electrical conversion'),
        ([380],'Listening means receiving information through hearing; speaking means communicating with your voice.','Written words are read, not listened to'),
        (range(384,388),'橡皮擦主要用来擦除纸上的铅笔痕迹，方便修改。','Do not imply ordinary erasers remove all inks'),
        (range(396,400),'A printer puts digital text or images onto paper.','Do not restrict all printing to liquid ink'),
        (range(420,424),'An empty box has nothing inside; a full box is filled to its capacity.','Nonempty is not the same as full'),
        (range(424,428),'借用前先征得物主同意，是尊重对方，也能避免误会。','Correct the omitted ownership/permission reason'),
        (range(468,472),'A fact is information supported by evidence; a personal preference describes what someone likes or dislikes.','Distinguish facts from statements that may be false'),
        (range(472,476),'标签能标明物品的名称、内容或类别，方便辨认和整理。','General labels are not limited to computer datasets'),
        ([231],'火箭发动机通过喷出气体产生推力，使火箭升空或改变速度。','Thrust changes motion; avoid implying continuous thrust is necessary to coast')]
    for indexes,text,reason in fixes:
        for i in indexes:
            name=f'teacher_{i}';overrides[name]=dict(original=data[name]['answer'],answer=text,reason=reason)
    e.save('teacher_curation.json',dict(excluded=excluded,overrides=overrides,raw_generations_preserved=True,
        fixed_sample='teacher_content_audit_sample.json;16 random accepted answers, seed202610275',
        content_inspection=dict(all160_simple_knowledge_answers_read=True,general_whitelist_count=len(inspected_general),fixed_supervision='560 programmatically constructed cases with known targets'),
        limit='Assistant content inspection and corrections; not independent expert factual verification.'))
    accepted=[r for k,r in data.items() if r['accepted'] and k not in excluded]
    print('Teacher retained',len(accepted),'excluded',len(excluded),'corrected',len(overrides),flush=True)
    for r in accepted:
        if r['source_kind']=='general':print(e.json.dumps(dict(id=r['id'],question=r['question'],answer=r['answer']),ensure_ascii=False))

if __name__=='__main__':main()
