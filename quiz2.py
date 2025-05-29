import pandas as pd

#한국_기업문화_HR_데이터셋_샘플.csv를 불러오시오.

df = pd.read_csv("한국_기업문화_HR_데이터셋_샘플.csv")

#데이터 전처리 (15점)

df = df.dropna(subset=["Age","이직여부","출장빈도","일일성과지표","부서","집까지거리","학력수준","전공분야","EmployeeCount","EmployeeNumber","근무환경만족도","성별","시간당급여","업무몰입도","직급","직무","업무만족도","결혼상태","월급여","MonthlyRate","이전회사경험수","Over18","야근여부","연봉인상률","성과등급","대인관계만족도","StandardHours","스톡옵션등급","총경력","연간교육횟수","워라밸","현회사근속년수","현직무근속년수","최근승진후경과년수","현상사근속년수"])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df['이직여부']] = scaler.fit_transform(df['이직여부'])

df_encoded = pd.get_dummies(df, columns=["Age","출장빈도","일일성과지표","부서","집까지거리","학력수준","전공분야","EmployeeCount","EmployeeNumber","근무환경만족도","성별","시간당급여","업무몰입도","직급","직무","업무만족도","결혼상태","월급여","MonthlyRate","이전회사경험수","Over18","야근여부","연봉인상률","성과등급","대인관계만족도","StandardHours","스톡옵션등급","총경력","연간교육횟수","워라밸","현회사근속년수","현직무근속년수","최근승진후경과년수","현상사근속년수"])

#이직 여부 이직 여부 예측에 유의미하다고 생각되는 피처 5~10개: "Age", "집까지거리", "근무환경만족도", "시간당급여", "직업만족도", "월급여", "성과등급", "연봉인상률", "워라밸"
#이유: 직장인들에게 가장 중요한 급여와 업무 환경과 관련된 변수들이다.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df_encoded.drop('이직여부', axis=1)
y = df_encoded['이직여부']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)








