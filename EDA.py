# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning) #Futurewarning 제거

train = pd.read_csv('data/train.csv')

# 데이터 기초 확인 작업
#print(train.head())
print(train.shape)

#결측치 확인
#결측치란, (NA : Not Avaialable) 값이 누락된 데이터를 말한다.
#정확한 분석을 위해서는 데이터 결측치를 확인하고 적절히 잘라서 처리해 주어야 한다.
def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)
print(missing_col)

#결측치가 존재하는 항목들이 범주형인지 수치형인지 확인
#print(train['workclass'].unique())
#print(train['occupation'].unique())
#print(train['native.country'].unique())

#결측치를 가진 모든 데이터가 범주형이기 때문에 범주형 데이터에 한해서는 행을 삭제해도 됨
def handle_na(data, missing_col):
    temp = data.copy()
    for col, dtype in missing_col:
        if dtype == 'O':
            # 범주형 feature가 결측치인 경우 해당 행들을 삭제해 주었습니다.
            temp = temp.dropna(subset=[col])
    return temp

train = handle_na(train, missing_col)
missing_col = check_missing_col(train)
print(missing_col)

#target을 소득 기준으로 재 정의
train['target'] = train['target'].apply(lambda x : '<=50K' if x == 0 else '>50K' )

#5만 달러 이하인지 초과인지 각각 클래스 분포 확인
counted_values = train['target'].value_counts()
plt.style.use('ggplot')
plt.figure(figsize = (12,10))
plt.title('class counting', fontsize = 30)
value_bar_ax = sns.barplot(x=counted_values.index, y=counted_values)
value_bar_ax.tick_params(labelsize=20)
plt.show()

print(train.info())

#범주형 피처 데이터를 시각화 하기 위해 범주형 피처만을 가진 데이터 프레임을 생성
train_categori = train.drop(['id', 'age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week'],axis = 1) #범주형이 아닌 피쳐 drop


