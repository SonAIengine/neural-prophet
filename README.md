# 먼저 라이브러리 설치부터
1. pip install -r requirements.txt

- get_stock_data : 데이터셋 만들기
- get_token : 한국증권API 토큰 발생

데이터 만들고싶으면
.env 파일에 아래 값 추가(한국증권 api 키 발급 받아야됨)
KOREA_INVESTMENT_APPKEY=*******************
KOREA_INVESTMENT_APPSECRET=*****************

만들어놓은 데이터셋 사용하려면
- merged_5_years_data_cleaned.csv

학습 코드
- predict_neuralprophet : Neural Prophet
- predict_prophet : Prophet
