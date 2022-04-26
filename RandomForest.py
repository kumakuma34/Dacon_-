import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

# csv 형식으로 된 데이터 파일을 읽어옵니다.
train = pd.read_csv('data/train.csv')
#결측치 확인.
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

#데이터셋을 train과 test로 나누기
data = train.drop('id',axis = 1).copy()
train_data, val_data = train_test_split(data,test_size=0.5)
train_data.reset_index(inplace=True)
val_data.reset_index(inplace=True)

X_train = train_data.drop(['index', 'target'], axis=1) #training 데이터에서 독립변수 추출
y_train = train_data.target #training 데이터에서 라벨 추출

model = RandomForestClassifier()
model.fit(X_train,y_train)

X_val = val_data.drop(['index', 'target'], axis=1)  #validation 데이터에서 전처리된 문서 추출
y_val = val_data.target #validation 데이터에서 라벨 추출

y_pred = model.predict(X_val)
print(y_pred)

print('RandomForestClassifier 의 예측 정확도는', round(metrics.accuracy_score(y_val, y_pred),3)) # 정확도 확인

#랜덤 포레스트 하이퍼 파라미터 튜닝
#1
#max_depth
model_md1 = RandomForestClassifier(max_depth = 1) # 모델을 객체에 할당
model_md10 = RandomForestClassifier(max_depth = 10)

model_md1.fit(X_train, y_train) # 모델 학습
model_md10.fit(X_train, y_train)

pred_md1 = model_md1.predict(X_val)
pred_md10 = model_md10.predict(X_val)

print('RF max_depth = 1 의 예측 정확도는', round(metrics.accuracy_score(y_val, pred_md1),3)) # 정확도 확인
print('RF max_depth = 10 의 예측 정확도는', round(metrics.accuracy_score(y_val, pred_md10),3)) # 정확도 확인

#2
#n_estimators
model_ne1 = RandomForestClassifier(n_estimators = 1) # 모델을 객체에 할당
model_ne200 = RandomForestClassifier(n_estimators = 200)

model_ne1.fit(X_train, y_train) # 모델 학습
model_ne200.fit(X_train, y_train)

pred_ne1 = model_ne1.predict(X_val)
pred_ne200 = model_ne200.predict(X_val)

print('RF n_estimators = 1 의 예측 정확도는', round(metrics.accuracy_score(y_val, pred_ne1),3)) # 정확도 확인
print('RF n_estimators = 200 의 예측 정확도는', round(metrics.accuracy_score(y_val, pred_ne200),3)) # 정확도 확인

#3
#max_features
model_mf1 = RandomForestClassifier(max_features = 1) # 모델을 객체에 할당
model_mf3 = RandomForestClassifier(max_features = 3)

model_mf1.fit(X_train, y_train) # 모델 학습
model_mf3.fit(X_train, y_train)

pred_mf1 = model_mf1.predict(X_val)
pred_mf3 = model_mf3.predict(X_val)

print('RF n_estimators = 1 의 예측 정확도는', round(metrics.accuracy_score(y_val, pred_mf1),3)) # 정확도 확인
print('RF n_estimators = 3 의 예측 정확도는', round(metrics.accuracy_score(y_val, pred_mf3),3)) # 정확도 확인

model_2 = RandomForestClassifier(max_depth = 10, n_estimators = 200, max_features = 3) # 최종 모델
model_2.fit(X_train, y_train)
y_pred_2 = model_2.predict(X_val)
print(y_pred_2)
print('Tuned RandomForestClassifier 의 예측 정확도는', round(metrics.accuracy_score(y_val, y_pred_2),3)) # 정확도 확인

#test data로 실행해보기
test = pd.read_csv('data/test.csv')
test = label_encoder(test, make_label_map(test))
test = test.drop('id', axis=1)

X_train= data.drop(['target'], axis=1) #전체 training 데이터에서 독립변수 추출
y_train = data.target #전체 training 데이터에서 라벨 추출

model = RandomForestClassifier(max_depth = 7, n_estimators = 200, max_features = 3) #앞서 튜닝한 하이퍼 파라미터 max_depth, n_estimators, max_features
model.fit(X_train, y_train)
y_pred = model.predict(test)

submission = pd.read_csv('data/sample_submission.csv')
submission['target'] = y_pred
submission.to_csv('submit_2.csv', index=False)
