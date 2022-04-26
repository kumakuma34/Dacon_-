import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

train = pd.read_csv('data/train.csv')

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
#print(missing_col)

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
#print(missing_col)

####데이터 전처리###

#Lable Encoding
def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            label_map = {'unknown':0}
            for i, key in enumerate(dataframe[col].unique()):
                label_map[key] = i  #새로 등장하는 유니크 값들에 대해 1부터 1씩 증가시켜 키값을 부여해줍니다.
            label_maps[col] = label_map
    return label_maps

#각 범주형 변수에 인코딩된 값을 부여
def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].map(label_map[col])
    return dataframe

train = label_encoder(train, make_label_map(train))

print(train)


#변수 및 모델 정의
X = train.drop(['id','target'], axis = 1)
Y = train['target']

model = LogisticRegression(solver='liblinear')
model.fit(X,Y)

def ACCURACY(true, pred):
    score = np.mean(true==pred)
    return score

prediction = model.predict(X)
score = ACCURACY(Y,prediction)

print(f"모델의 정확도는 {score*100:.2f}% 입니다")


#test 파일로 실행
test = pd.read_csv('data/test.csv')

test = label_encoder(test,make_label_map(test))
test = test.drop(['id'], axis = 1)
prediction = model.predict(test)
print(prediction)

submission = pd.read_csv('data/sample_submission.csv')
submission['target'] = prediction

print(submission)

submission.to_csv('submit.csv',index=False)
