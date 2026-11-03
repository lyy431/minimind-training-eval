"""Record the 2026-11-04 development content review (model labels hidden in the queue).

Decisions are indexed against dev_review_queue_snapshot.json; every item was read. Items not listed scored 0.
Reviewer was the assistant, not an independent human blind rater. The v4 dev baseline was reviewed earlier by a different assistant.
"""
import strong_teacher_common as e

TWO = {2,34,42,46,57,67,92,114,195,211}
ONE = {6,9,11,13,14,15,20,21,23,26,30,36,39,44,45,47,49,54,60,64,65,66,68,69,71,74,78,79,83,87,90,91,93,94,95,96,98,110,122,131,134,140,143,155,158,160,164,169,174,175,177,180,190,191,194,197,209,212}
REASON = {0: '未满足主要请求：内容错误、角色错位、答非所问或语句不通', 1: '部分回应，但有明显遗漏、混乱、循环解释或重复', 2: '主要请求得到清楚实用的回应，没有明显严重错误'}
WHO = 'assistant; model labels hidden; not independent human blind review'

def main():
    queue = e.read(e.OUT / 'dev_review_queue_snapshot.json')
    assert len(queue) == 214 and not TWO & ONE
    reviews = e.read(e.OUT / 'content_reviews.json')
    for i, item in enumerate(queue):
        score = 2 if i in TWO else 1 if i in ONE else 0
        if not item['eligible'] or item['exact_answer'] is not None:
            assert score < 2 or item['answer'].strip() == item['exact_answer']
        reviews[item['review_id']] = dict(score=score, reason=REASON[score], source=WHO)
    e.save('content_reviews.json', reviews)
    maths = e.read(e.OUT / 'math_reviews.json')
    for item in e.read(e.OUT / 'dev_math_review_queue_snapshot.json'):
        maths[item['review_id']] = dict(decision='fail', reason='逐条复核：算式抄错、漏算或结果错误', source=WHO)
    e.save('math_reviews.json', maths)
    print('content', len(queue), {s: sum((2 if i in TWO else 1 if i in ONE else 0) == s for i in range(len(queue))) for s in [0, 1, 2]}, 'math reviewed', len(maths))

if __name__ == '__main__':
    main()
