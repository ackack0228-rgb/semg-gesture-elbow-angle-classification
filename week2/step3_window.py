# Step 2 · 2주차 실습 · 단계 3 : 슬라이딩 윈도우 (제출물 ② 윈도우 배열 모양)
# 목적: 2초 세그먼트(2,000샘플)를 200 ms 길이로 100 ms씩 밀며 잘라 19개 조각으로 만든다.
#
# 실행 방법: python step3_window.py   (step2_segment.py 를 먼저 실행)
# 결과: step3_w.npy (다음 단계 입력)

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WIN, HOP = 200, 100            # 윈도우 200 ms, 증분 100 ms (50% 오버랩) — 1 kHz라서 샘플 수 = ms


def make_windows(x, win=WIN, hop=HOP):
    # (2000, 5) -> (19, 200, 5).  조각 수 = (길이 - 윈도우) // 증분 + 1
    n = (len(x) - win) // hop + 1
    return np.stack([x[i * hop: i * hop + win] for i in range(n)])


if __name__ == '__main__':
    seg = np.load(os.path.join(BASE_DIR, 'step2_seg.npy'))
    w = make_windows(seg)
    print('세그먼트 모양:', seg.shape)
    print('윈도우 배열:', w.shape)      # (19, 200, 5)
    print('계산: (2000 - 200) / 100 + 1 =', (len(seg) - WIN) // HOP + 1)
    # 이웃한 조각은 100샘플(50%)이 똑같다 — 그래서 같은 시행의 조각이 학습·테스트에 나뉘면 누수가 된다(단계 5).
    print('1번 조각의 뒤 100샘플 == 2번 조각의 앞 100샘플:', np.array_equal(w[0, HOP:], w[1, :HOP]))
    print('조각이 덮는 범위: 0 ~', (len(w) - 1) * HOP + WIN, '샘플 (세그먼트 전체)')
    np.save(os.path.join(BASE_DIR, 'step3_w.npy'), w)
    print('저장 완료: step3_w.npy')
