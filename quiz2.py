import pandas as pd

#한국_기업문화_HR_데이터셋_샘플.csv를 불러오시오.

df = pd.read_csv("한국_기업문화_HR_데이터셋_샘플.csv")

#데이터 전처리 (15점)

df = df.dropna(subset=["Age","이직여부","출장빈도","일일성과지표","부서","집까지거리","학력수준","전공분야","EmployeeCount","EmployeeNumber","근무환경만족도","성별","시간당급여","업무몰입도","직급","직무","업무만족도","결혼상태","월급여","MonthlyRate","이전회사경험수","Over18","야근여부","연봉인상률","성과등급","대인관계만족도","StandardHours","스톡옵션등급","총경력","연간교육횟수","워라밸","현회사근속년수","현직무근속년수","최근승진후경과년수","현상사근속년수"])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df['이직여부']] = scaler.fit_transform(df['이직여부'])

df_encoded = pd.get_dummies(df, columns=["Age","출장빈도","일일성과지표","부서","집까지거리","학력수준","전공분야","EmployeeCount","EmployeeNumber","근무환경만족도","성별","시간당급여","업무몰입도","직급","직무","업무만족도","결혼상태","월급여","MonthlyRate","이전회사경험수","Over18","야근여부","연봉인상률","성과등급","대인관계만족도","StandardHours","스톡옵션등급","총경력","연간교육횟수","워라밸","현회사근속년수","현직무근속년수","최근승진후경과년수","현상사근속년수"])

#피처 선택 (15점)

#이직 여부 이직 여부 예측에 유의미하다고 생각되는 피처 5~10개: "Age", "집까지거리", "근무환경만족도", "시간당급여", "직업만족도", "월급여", "성과등급", "연봉인상률", "워라밸"
#이유: 직장인들에게 가장 중요한 급여와 업무 환경과 관련된 변수들이다.

features = ["Age", "집까지거리", "근무환경만족도", "시간당급여", 
            "직업만족도", "월급여", "성과등급", "연봉인상률", "워라밸"]

X = df[features]
y = df['이직여부']



#모델 훈련 (20점)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

#성능 검증 (10점)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
print("혼동 행렬:")
print(cm)


#예측 결과 분석 (10점)
y_prob = model.predict_proba(X_test)[:, 1]

yes_count = sum(y_pred)
print(f"Yes 직원 수: {yes_count}")

result_df = X_test.copy()
result_df['이직확률'] = y_prob
top_5 = result_df.sort_values(by='이직확률', ascending=False).head(5)

#신입사원 예측 (10점)


신입사원_데이터 = [{
    "Age": 29, "DistanceFromHome": 5, "EnvironmentSatisfaction": 2,
    "HourlyRate": 70, "JobSatisfaction": 2, "MonthlyIncome": 2800,
    "PerformanceRating": 3, "PercentSalaryHike": 12, "WorkLifeBalance": 2
    },
    {
        "Age": 42, "DistanceFromHome": 10, "EnvironmentSatisfaction": 3,
        "HourlyRate": 85, "JobSatisfaction": 4, "MonthlyIncome": 5200,
        "PerformanceRating": 3, "PercentSalaryHike": 14, "WorkLifeBalance": 3
    },
    {
        "Age": 35, "DistanceFromHome": 2, "EnvironmentSatisfaction": 1,
        "HourlyRate": 65, "JobSatisfaction": 1, "MonthlyIncome": 3300,
        "PerformanceRating": 3, "PercentSalaryHike": 11, "WorkLifeBalance": 2
    }
]

new_predictions = model.predict(df_new_employees)
for i, pred in enumerate(new_predictions, 1):
    print(f"신입사원 {i} 이직 예측: {pred}")

#다음 중 한 가지 방법을 사용하여, **이직 여부 예측에 가장 큰 영향을 준 피처(컬럼)**를 파악하고 결과를 해석하시오: (15점)

② 랜덤 포레스트 사용 시

# 모델 훈련 후
importances = model.feature_importances_
features = X_train.columns
importance = pd.Series(importances, index=features).sort_values(ascending=False)



