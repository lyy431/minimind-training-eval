"""Record the 2026-11-04 final content and math review (model labels hidden in the queue).

Decisions are indexed against final_review_queue_snapshot.json; every item was read. Items not listed scored 0.
Reviewer was the assistant, not an independent human blind rater.
"""
import strong_teacher_common as e

TWO = {39,44,63,65,106,112,141,146,168,247,262,274,301,314,316,323,328,358,361,370,384,397,406,421,448}
ONE = {1,18,20,22,35,36,49,50,52,53,56,58,59,68,75,79,81,97,98,100,102,115,120,123,125,130,137,140,152,180,186,189,197,202,206,210,214,
       215,217,223,232,235,257,265,267,270,293,294,295,299,302,307,309,310,315,319,321,331,347,353,360,369,377,387,405,412,416,417,427,430,431,432,439,442,451}
REASON = {0: '未满足主要请求：内容错误、角色错位、答非所问或语句不通', 1: '部分回应，但有明显遗漏、混乱、语言不符或格式不符', 2: '主要请求得到清楚实用的回应，没有明显严重错误'}
MATH_PASS = {4, 9}
WHO = 'assistant; model labels hidden; not independent human blind review'

def main():
    queue = e.read(e.OUT / 'final_review_queue_snapshot.json')
    maths = e.read(e.OUT / 'final_math_review_queue_snapshot.json')
    assert len(queue) == 453 and len(maths) == 15 and not TWO & ONE
    reviews = e.read(e.OUT / 'content_reviews.json')
    for i, item in enumerate(queue):
        score = 2 if i in TWO else 1 if i in ONE else 0
        assert not (score == 2 and (not item['eligible'] or item['exact_answer'] is not None))
        reviews[item['review_id']] = dict(score=score, reason=REASON[score], source=WHO)
    e.save('content_reviews.json', reviews)
    decisions = e.read(e.OUT / 'math_reviews.json')
    for i, item in enumerate(maths):
        passed = i in MATH_PASS
        decisions[item['review_id']] = dict(decision='pass' if passed else 'fail',
            reason='逐条核对：全部结果正确且对应正确算式' if passed else '逐条复核：漏答、错配、结果错误或表述矛盾', source=WHO)
    e.save('math_reviews.json', decisions)
    print('content', len(queue), 'two', len(TWO), 'one', len(ONE), 'zero', len(queue) - len(TWO) - len(ONE), 'math pass', len(MATH_PASS), 'of', len(maths))

if __name__ == '__main__':
    main()
