### train_data

feature : 

```
PassengerId : row pk
Survived : 생존여부 1 or 0
Pclass : 티켓의 클래스 1,2,3
Name : 승객 이름 > 정규표현식을 이용하여 Initial 컬럼으로 Mr,Miss 등을 구분
Sex : 성별
Age : 나이 > 나중에 연령대로 구분
SibSp : 함께 탑승한 형제 or 배우자 수
Parch : 함께 탑승한 부모, 아이 수
Ticket : 티켓 번호 (Alphabet + Int)
Fare : 탑승료     
Cabin : 객실번호 (Alphavet + Int)
Embarked : 탑승 하구 C,Q,S
```

결과 Target : `Survived`



1. null이 들어 있는 feature 파악

2. data analysis : 시각화를 통한 데이터의 사실 정보 파악

   1) target이 생존여부이므로 각 생존율을 알 수 있는 feature을 시각화

   >Pclass  - Survived / Sex - Survived / Sex - Plcass vs Survived(남녀의 pclass 별 생존율) /
   >
   >Age - Survived / Pclass-Age vs Survived / Sex-Age vs Survived / Embarked - Survived /
   >
   >Embarked-Sex vs Survived / Embarked-Pclass vs Survived

   >생존율과 데이터의 갯수의 순서는 다를 수 있음. 에로 Embarked는 S에 위치한 승객의 수가 가장많다. S에서 생존한 사람의 수는 가장 많다. 하지만 전체 인원 중 C에 위치한 승객의 생존 범위가 더 높아 생존율은 C가 더 높다.

   >  SibSp+Parch +1(본인) = Familysize로 총 가족수로 count
   >
   > 가족수 별 생존율 : FamilySize - Survived
   >
   > Fare 요금 종류가 다른게 248개가 있고 0.0원짜리도있음. -> 비대칭 심해 log로 완만하게 함

3. 전처리

   1) null 값 채우기

   > * Name을 통해서 유추할 수 있는 String을 변환 > AGE의 null 값 채우는데 활용
   >
   > `df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.')` 정규표현식으로
   >
   > `['Mile','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona']`로 분류함 그리고 이 공통적인 부분을 다시
   >
   > `[Master,Miss,Mr,Mrs,Others]`로 Category로 만듦(Initial 별로 성별을 각각 얼마나 가지고 있는지 count하여 많은 쪽에 통일)  > Encoding하여 feature로 활용
   >
   > 이로써 새로 생성된 Initial은 Age와 함께 null을 채우는 지표로 사용 Ex) age가 null이고 Initial이 Mr이면 33세로 값을 대입 
   >
   > ​	(숫자는`df_train.groupby('Initial').mean()`으로 Age가 Initial마다 가지고 있는 평균 AGE가 몇인지 알 수 있음)
   >
   > * Embarked null값 채우기
   >
   > S,C,Q로 되어 있고 null값이 2개밖에 없으므로 value는 가장 많이 포함하고 있는 S로 통일
   >
   > * categorical하게 변환
   >
   > Age를 카테고리화 하여 연령대로 바꾸고 이를 0-5번으로 나눈다. 10대미만 : 0, 10대:1 ....

   2) Encoder

   > String을 Integer로 변환
   >
   > * 하드코드
   >
   > 카테고리로 되어 있는 Initial을 ``[Master,Miss,Mr,Mrs,Others]`` 에서 Int로 변환한다.
   >
   > Embarked 또한`[S,C,Q]`에서 Int로 변환한다.
   >
   > Sex 또한 [Male,Female]에서 Int로 변환한다.
   >
   > ```
   > df_train['Initial'] = df_train['Initial'].map({
   >     'Master' : 0,
   >     'Miss' : 1,
   >     'Mr' : 2,
   >     'Mrs' : 3,
   >     'Other': 4
   > })
   > 
   > df_train['Embarked'] = df_train['Embarked'].map({
   >     'C' : 0,
   >     'Q' : 1,
   >     'S' : 2
   > })
   > df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
   > ```
   >
   > * one-hot Encoding
   >
   > 1)하드코딩으로 int변환한 Initial을 ont-hot 인코딩으로 카테고리만큼 컬럼을 생성한 matrix형태로 만든다. 행렬 위치에 값이 존재하면 1을 넣어 0 또는 1의 binary로 만든다.
   >
   > 2)하드코딩으로 int변환한 Embarked 또한 one-hot 인코딩으로 nxn의 matrix로 만든다.
   >
   > 역시나 binary로 표현되게 함.
   >
   > : 즉, 카테고리로 묶을 수 있는 string을 0-2이상으로 변환될 시 binary로 보여주기 위해 one-hot encoding으로 사용
   >
   > * label-encoding
   >
   > Ticket value을 labelencoder하여 string을 고유 int값으로 변환한다.
   >
   > 갯수가 681개
   >
   > * Word2vec
   >
   > one-hot 인코딩 같이 카테고리관계를 유지한 인코딩이 좋은데 범주가 많으면 비효율적이므로 범주가 많은 경우에는 Word2vec라이브러리를 사용한다.

   3) 필요없는 컬럼 지우기

   > 모델의 학습 정확도를 위해 필요없는 Feature을 제거한다.
   >
   > `df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)`

4. 모델 학습

   1) 모델을 선정

   > Titanic에서는 
   >
   > RandomForestClassifier, LogisticRegession, KNN, GradientBoostingClassifier

   2) training data(input, output), valid data(input, output), test data 분류

   > test data가 없을 시 training  data에서 일부 분리(target제외)
   >
   > training은 학습용, valid는  training data와 같고 로컬의 output과 학습한 모델에서 넣은 output과 비교하여 정확도를 도출한다.
   >
   > `x_tr,x_vld,y_tr,y_vld = train_test_split(INPUT OBJECTS,OUTPUT OBJECT,test_size=0.3,random_state=2018)`

   3) model.fit

   4) model.predict로 valid input을 넣어 결과 도출

   5) valid output과 prediction결과를 비교하여 정확도 보기

   6) test data를 넣어서 결과가 어떻게 나오는지 보기

5. Elaborating

   1) data 구조에 맞는 최적화 알고리즘 

   2) 데이터의 사실 정보를 보고 전처리 과정 바꾸기

   3) 데이터의 사실 정보를 통해 특징을 만들어 새로운 feature를 만들고 학습