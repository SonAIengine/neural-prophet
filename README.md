# 해보려면

## 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

## 주요 기능

- **get_stock_data**: 데이터셋 생성 기능
- **get_token**: 한국증권 API 토큰 생성 기능

## 데이터 생성 방법

데이터를 생성하려면 `.env` 파일에 아래 내용을 추가해야 합니다. 한국증권 API 키를 발급받아 `APPKEY`와 `APPSECRET` 값을 입력하세요.

```env
KOREA_INVESTMENT_APPKEY=*******************
KOREA_INVESTMENT_APPSECRET=*****************
```

## 기존 데이터셋 사용

만들어놓은 데이터셋 사용하려면(삼성전자 5년치) / 현재 학습코드는 해당 csv파일을 가지고 하는걸로 되어있음.

- **merged_5_years_data_cleaned.csv**

## 학습 코드 실행

### Neural Prophet 모델을 사용한 예측

```bash
python predict_neuralprophet.py
```

### Prophet 모델을 사용한 예측

```bash
python predict_prophet.py
```

