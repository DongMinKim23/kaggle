>data analysis model 이해 및 역량 강화
>
>titanic data 활용 - for starter -> 목 or 금 중 프레젠테이션
>
>DS 주간보고 양식 1주차 실적 반영 및 다음주 계획 작성 목업



Kaggle-Titanic



데이터 예측 순서

1. train용 data와 test용 data를 분리(보통은 train과 data 사이에 valid 단계가 있어 valid를 통한 acuuracy 평가, train과 test용은 같은 인풋이 들어가도록 전처리 모두 동일)

2. data의 feature 들을 살펴본다.

   1. null 값이 들어있는지

   2. feature1과 예측 결과를 얻어야하는 feature2가 user id마다 어떤 결과를 보여주는지 grouping 

      > ex) 타이타닉의 경우
      >
      > Pclass(좌석등급)과 Survived의 관계를 알아보고, 등급이 높거나 낮을 수록 생존율이 높다는 것을 볼 수 있다면 이를 활용하여 생존율을 예측할 Inputdata로 여기면 된다.
      >
      > Pclss를 연령대, 성별 등으로 기존의 data를 가지고 예측값과의 관계를 그래프 또는 비율로 만들어 Input으로 활용할 컬럼과 그렇지 않은 컬럼을 구분한다.

3. 한 feature가 담고 있는 value가 conitnuous value라고 한다면 이를 히스토그램과 같은 그래프로 차이가 있는지 본다. 만약 차이가 있다면 이를 완만하게 만든다.

4. Null값을 채운다. **IDEA

   1. 연령대와 같이 contiuously한 value일 경우, 범위를 만든다.(남자이고 직업이 어느 경우일 경우 나이가 30대가 많다? 그럼 30대로 , data를 string으로 바꿔주는 정규표현식을 이용해 직업이 없는 경우 직업과 같은 feature을 만들어서 연령대 범위를 정할 수도 있다.)
   2.  직업이나 호칭 mr, mrs,miss,doctor 과 같은 것들은 0,1,2,3,4와 같이 등급을 숫자로 표현해준다.
   3. A,B,C와 같이 등급으로 된 feature는 가장 많은 value를 조회하고 그 value로 null을 채울 수도 있다.

5. String으로 된 것은 컴퓨터가 인식할 수 있는 binary로 표현하거나 어떤 숫자로 변형한다. **IDEA

   (만약 string을 활용할 방법이 생각이 나면 그 방향으로 숫자로 바꿔서 해도 됨.)

   1. 그냥 코드

   2. one-hot-encoding

      string으로 된 카테고리만큼 컬럼을 만들어 거기에 해당하는 값의 컬럼에 1을 준다. matrix로 표현

   3. word2vec

6. 학습 모델에 필요한 컬럼은 남기고 나머진 지운다.

7. 학습 모델을 만든다 **IDEA

   1. 학습 모델은 python 내장 패키지를 이용
   2. train data에서 x(input)과 y(output)을 구분
   3. model을 만들고 input을 넣어본다.
   4. input을 넣어본 예측값을 실제 train data에 명시된 output값과 비교한다. 
   5. acurracy를 보고 너무 낮으면 모델의 파라미터 튜닝 작업을 해야한다.

8. 각 모델의 성능 또는 모델이 지니고 있는 feature 중에 어떤 feature가 해당 모델에 영향을 크게 주고 있는지도 보면 좋겠다.

9. 정확도가 어느정도 나오면 test data를 넣어 결과를 본다.



randomforest 82% / ticket label encoder추가 85%(feature importance가 ticket이 됨.)

logistic 86% /  ''같은 조건"" 87% 됨

knn 82% / 72%나 감소됨....

gradient boosting 86% / 85%로 감소됨





`models 중 logistic regression이 제일 정확도가 좋음`

`ticket을 label encoder하니 증가폭은 randomforest가 가장 많음. logistic은 1%증가`





