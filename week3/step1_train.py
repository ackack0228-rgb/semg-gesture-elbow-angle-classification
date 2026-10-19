# Step 2 · 3주차 실습 · 단계 1 : 세 모델 학습
# 목적: SVM · RF · XGBoost를 같은 데이터 · 같은 분할(2주차 시행 단위 8:2)로 학습한다.
#       표본 구성 A · B · C (common3.py 설명) 마다 세 모델을 학습해 모두 9개 모델을 만든다.
#
# 실행 방법: python step1_train.py   (2주차 step5_dataset.py · step6_scale_save.py 를 먼저 실행)
# 결과: models/A_SVM.pkl ... models/C_XGB.pkl  (scaler와 모델을 함께 저장, 단계 2·3에서 불러 씀)

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from common3 import (WEEK2_DIR, COMPS, COMP_NAME, load_holdout, make_models, represent, fit,
                     save_model)

(Ta, ya, ga, sa), (Tb, yb, gb, sb) = load_holdout()
# 라벨은 2주차에 이미 0~7 정수로 저장했지만 슬라이드대로 LabelEncoder를 거친다
# (XGBoost는 0부터 시작하는 연속 정수 라벨이 필요 — 슬라이드 16쪽 첫 번째 오류).
le = LabelEncoder(); ya = le.fit_transform(ya)
yb = le.transform(yb)
print('학습 시행:', len(ya), ' 테스트 시행:', len(yb), ' 클래스:', le.classes_.tolist())

# 확인: 구성 A의 표준화 결과가 2주차 dataset.npz 와 같은지 (같은 데이터 · 같은 분할 · 같은 표준화)
d = np.load(os.path.join(WEEK2_DIR, 'dataset.npz'))
sc_a, _, _ = fit('A', make_models()['RF'].set_params(n_estimators=1), Ta, ya)
print('구성 A = dataset.npz (학습 · 테스트):',
      np.allclose(sc_a.transform(represent(Ta, 'A')), d['Xtr']),
      np.allclose(sc_a.transform(represent(Tb, 'A')), d['Xte']))

for comp in COMPS:
    print(f'\n[{COMP_NAME[comp]}]  학습 표본 모양:', represent(Ta, comp).shape)
    for k, m in make_models().items():
        sc, m, t = fit(comp, m, Ta, ya)
        print(f'  {k} 학습 완료 ({t:.1f}초)')
        save_model(f'{comp}_{k}', {'scaler': sc, 'model': m})

# 재현성 확인: 같은 설정으로 다시 학습하면 예측이 완전히 같아야 한다 (random_state=42)
from common3 import predict, load_model
for comp in COMPS:
    saved = load_model(f'{comp}_RF')
    sc, m, _ = fit(comp, make_models()['RF'], Ta, ya)
    same = (predict(comp, sc, m, Tb)[0] == predict(comp, saved['scaler'], saved['model'], Tb)[0]).all()
    print(f'{comp} RF 재학습 예측 일치:', bool(same))
