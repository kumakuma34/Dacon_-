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
#plt.show()

print(train.info())

#범주형 피처 데이터를 시각화 하기 위해 범주형 피처만을 가진 데이터 프레임을 생성
train_categori = train.drop(['id', 'age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week'],axis = 1) #범주형이 아닌 피쳐 drop

#범주형 데이터 분포를 확인해보자
def visualize(axx, field, num): ##그래프를 그리기 위한 메소드
    sns.countplot(train_categori.columns[num], data= train_categori[train_categori['target'] == field],  color='#eaa18a', ax = axx) # countplot을 이용하여 그래프를 그려줍니다.
    axx.set_title(field)

figure, ((ax1,ax2),(ax3,ax4), (ax5, ax6),(ax7, ax8), (ax9, ax10),
         (ax11,ax12),(ax13,ax14), (ax15, ax16))  = plt.subplots(nrows=8, ncols=2) ## 원하는 개수의 subplots 만들어주기
figure.set_size_inches(40, 50) #(w,h)
figure.suptitle('Compare categorical features', fontsize=40, y = 0.9)

k = 0 # 피쳐 수
j = 1 # 그래프 수
while k<8:
    for i in range(0,2):
        visualize(eval(f'ax{j}'), train_categori['target'].unique()[i], k)
        j = j+1
    k = k+1

plt.show()


# 수치형 데이터 분포 확인
train_numeric = train[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week', 'target']] #수치형 피쳐와 label인 target 추출

def visualize(axx, field, num):
    line = train_numeric[train_numeric['target'] == field] #메소드에서 target 클래스 추춣
    name = train_numeric[train_numeric['target'] == field][train_numeric.columns[num]].name #메소드에서 이름 추출
    sns.kdeplot(x = line[train_numeric.columns[num]],  data = train_numeric, ax = axx, color='#eaa18a') #countplot을 이용하여 그래프를 그려줍니다.
    axx.axvline(line.describe()[name]['mean'], c='#f55354', label = f"mean = {round(line.describe()[name]['mean'], 2)}") #mean 통계값을 표기해줍니다.
    axx.axvline(line.describe()[name]['50%'], c='#518d7d', label = f"median = {round(line.describe()[name]['50%'], 2)}") #median 통계값을 표기해줍니다.
    axx.legend()
    axx.set_title(field)

figure, ((ax1,ax2),(ax3,ax4), (ax5, ax6),(ax7, ax8), (ax9, ax10))  = plt.subplots(nrows=5, ncols=2) ##원하는 개수의 subplots 만들어주기
figure.set_size_inches(40, 50) #(w,h)
figure.suptitle('Compare numeric features', fontsize=40, y = 0.9)

k = 0 # 피쳐 수
j = 1 # 그래프 수
while k<5:
    for i in range(0,2):
        visualize(eval(f'ax{j}'), train_numeric['target'].unique()[i], k)
        j = j+1
    k = k+1

plt.show(block=True)


#상관관계
plt.style.use('ggplot')
plt.figure(figsize=(12, 10))
plt.title('capital gain and working time', fontsize = 30)
sns.scatterplot(x = 'capital.gain',  y= 'hours.per.week', hue= 'target', data= train[train['capital.gain'] > 0]) #산포도를 확실하게 차이나도록  시각화 해주기 위하여 capital.gain에서 0값을 제외

plt.style.use('ggplot')
plt.figure(figsize=(12, 10))
plt.title('capital gain and working time', fontsize = 30)
sns.scatterplot(x = 'age',  y= 'capital.loss', hue= 'target', data= train[train['capital.loss'] > 0]) #산포도를 확실하게 차이나도록  시각화 해주기 위하여 capital.loss에서 0값을 제외