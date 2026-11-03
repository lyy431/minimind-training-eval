"""Record the 2026-11-04 open-answer teacher review (teacher labels hidden in the queue).

Decisions are indexed against teacher_review_queue_snapshot.json, the frozen copy of the 998-item queue. Every item
was read; items not listed scored 2. Reviewer was the assistant, not an independent human blind rater.
"""
import json
import strong_teacher_common as e

REASONS = {
    'R': '拒绝完成请求',
    'G': '夹杂其他语言或乱码，不能直接使用',
    'P': '人称或方向写反，没有完成用户要发送的内容',
    'W': '内容与请求不符或含明显错误',
    'M': '含说明性前缀、称呼占位符或助手口吻，不能直接发送',
    'I': '编造了未提供的具体姓名、时间或地点',
    'L': '超过要求长度或使用长列表，不符合简短示范要求',
    'O': '没有说清核心原因，内容泛泛或偏题',
    'A': '遗漏请求中的关键信息',
    'Z': '回答语言与要求不符',
}
ZERO = {5:'R',21:'G',25:'P',44:'W',53:'G',61:'P',62:'P',83:'W',84:'P',88:'W',105:'W',106:'P',110:'P',139:'W',150:'W',180:'Z',195:'P',
    209:'R',243:'W',255:'P',257:'W',307:'W',341:'P',342:'W',352:'P',367:'I',369:'M',
    436:'P',465:'W',475:'W',480:'P',494:'W',508:'Z',509:'P',535:'Z',567:'P',578:'W',581:'Z',587:'M',
    603:'P',607:'W',646:'P',694:'P',698:'W',703:'P',716:'W',738:'W',754:'W',759:'W',776:'W',791:'P',
    801:'P',804:'P',805:'W',806:'R',810:'W',814:'P',826:'G',838:'P',858:'P',889:'W',904:'W',908:'P',909:'W',912:'P',987:'W'}
ONE = {4:'M',8:'L',15:'L',17:'O',29:'L',37:'A',41:'A',48:'I',54:'A',65:'W',87:'W',90:'I',101:'G',109:'P',121:'M',123:'M',124:'L',128:'I',
    132:'L',153:'W',158:'L',159:'I',160:'G',182:'W',183:'L',186:'M',189:'L',193:'W',197:'M',
    206:'A',217:'L',227:'L',231:'O',260:'O',261:'L',264:'W',266:'W',268:'L',275:'L',278:'M',282:'W',284:'L',287:'M',300:'W',304:'M',305:'M',
    310:'O',313:'M',315:'M',320:'L',324:'I',325:'G',337:'W',348:'W',355:'M',356:'L',357:'L',361:'M',374:'A',385:'W',386:'I',390:'L',392:'L',393:'M',395:'M',
    406:'A',408:'W',430:'W',435:'L',440:'I',442:'W',445:'A',446:'O',448:'W',454:'O',462:'W',468:'A',473:'A',504:'O',505:'M',515:'W',518:'L',519:'P',
    534:'I',537:'O',538:'L',544:'L',546:'W',547:'M',562:'L',564:'W',570:'L',575:'M',584:'M',585:'M',596:'I',599:'M',
    608:'A',615:'L',616:'L',619:'A',628:'W',632:'M',636:'M',651:'M',652:'A',656:'L',658:'W',663:'M',691:'I',695:'W',699:'M',709:'L',721:'O',722:'M',725:'W',
    751:'I',756:'G',763:'A',779:'W',781:'W',785:'W',792:'L',793:'M',797:'M',798:'M',799:'I',
    807:'P',830:'O',837:'W',866:'M',871:'L',881:'L',886:'M',898:'O',899:'W',910:'L',913:'L',915:'G',918:'O',924:'W',931:'L',938:'I',940:'L',943:'M',944:'M',945:'A',
    953:'O',961:'O',965:'M',970:'M',971:'W',973:'L',977:'L',978:'L',992:'L',995:'M'}
PASS = {'writing': '可以直接发送，语言和内容符合请求', 'explanation': '解释准确、简短，回答了问题'}

def main():
    snapshot = e.OUT / 'teacher_review_queue_snapshot.json'
    assert snapshot.exists()
    queue = e.read(snapshot)
    assert len(queue) == 998 and not set(ZERO) & set(ONE) and max(max(ZERO), max(ONE)) < len(queue)
    reviews = {}
    for i, item in enumerate(queue):
        if i in ZERO:
            reviews[item['review_id']] = dict(score=0, reason=REASONS[ZERO[i]], reviewer='assistant; teacher labels hidden; not independent human blind review')
        elif i in ONE:
            reviews[item['review_id']] = dict(score=1, reason=REASONS[ONE[i]], reviewer='assistant; teacher labels hidden; not independent human blind review')
        else:
            reviews[item['review_id']] = dict(score=2, reason=PASS[item['category']], reviewer='assistant; teacher labels hidden; not independent human blind review')
    assert len(reviews) == 998
    e.save('teacher_content_reviews.json', reviews)
    print('recorded', len(reviews), 'score0', len(ZERO), 'score1', len(ONE), 'score2', 998 - len(ZERO) - len(ONE))

if __name__ == '__main__':
    main()
