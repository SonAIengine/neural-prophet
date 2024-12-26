import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd

# .env 파일 로드
load_dotenv()

# 토큰 저장 경로 설정
TOKEN_FILE = "access_token.json"


def get_access_token(appkey, appsecret, is_mock=True):
    """
    접근 토큰을 캐싱하여 관리하며, 필요 시 새로 발급.
    """
    # 1. 캐시된 토큰 확인
    token_data = load_cached_token()
    if token_data:
        access_token = token_data.get("access_token")
        expires_at = token_data.get("expires_at")

        # 토큰이 유효하면 그대로 사용
        if access_token and expires_at:
            expires_at = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
            if expires_at > datetime.now():
                print("Cached Access Token 사용")
                return access_token

    # 2. 새 토큰 발급
    print("새로운 Access Token 발급 요청")
    token_response = request_new_token(appkey, appsecret, is_mock)
    if "access_token" in token_response:
        # 유효 기간 설정 (현재 시간 + expires_in 초)
        expires_in = token_response.get("expires_in", 86400)  # 기본 24시간
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        # 토큰 저장
        access_token = token_response["access_token"]
        save_cached_token(access_token, expires_at)

        return access_token
    else:
        print("Error:", token_response.get("message"))
        return None


def request_new_token(appkey, appsecret, is_mock=True):
    """
    새로운 접근 토큰 발급 요청
    """
    if is_mock:
        url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"  # 모의 환경
    else:
        url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"  # 실전 환경

    payload = {
        "grant_type": "client_credentials",
        "appkey": appkey,
        "appsecret": appsecret,
    }

    headers = {"Content-Type": "application/json; charset=UTF-8"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code, "message": response.text}
    except requests.RequestException as e:
        return {"error": "RequestException", "message": str(e)}


def save_cached_token(access_token, expires_at):
    """
    캐시된 토큰을 파일에 저장
    """
    token_data = {
        "access_token": access_token,
        "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(TOKEN_FILE, "w") as file:
        json.dump(token_data, file)
    print("Access Token 저장 완료")


def load_cached_token():
    """
    캐시된 토큰을 파일에서 로드
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file:
            return json.load(file)
    return None


def get_financial_ratio(access_token, appkey, appsecret, stock_code, div_code="0"):
    """
    재무비율 정보를 가져오는 함수

    Parameters:
    - access_token: API 접근 토큰
    - appkey: 앱키
    - appsecret: 앱시크릿
    - stock_code: 종목코드
    - div_code: 분류 구분 코드 (0: 년, 1: 분기)

    Returns:
    - DataFrame: 재무비율 데이터
    """
    url = "https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/finance/financial-ratio"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": "FHKST66430300",
    }

    params = {
        "FID_DIV_CLS_CODE": div_code,  # 0: 년, 1: 분기
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code,
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()

            if "output" in data:
                df = pd.DataFrame(data["output"])

                # 컬럼명을 한글로 변경
                column_mapping = {
                    "stac_yymm": "결산년월",
                    "grs": "매출액증가율",
                    "bsop_prfi_inrt": "영업이익증가율",
                    "ntin_inrt": "순이익증가율",
                    "roe_val": "ROE",
                    "eps": "EPS",
                    "sps": "주당매출액",
                    "bps": "BPS",
                    "rsrv_rate": "유보율",
                    "lblt_rate": "부채비율",
                }
                df = df.rename(columns=column_mapping)

                # 데이터 타입 변환
                numeric_columns = [
                    "매출액증가율",
                    "영업이익증가율",
                    "순이익증가율",
                    "ROE",
                    "EPS",
                    "주당매출액",
                    "BPS",
                    "유보율",
                    "부채비율",
                ]
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # 결산년월 형식 변환 (YYYYMM -> datetime)
                df["결산년월"] = pd.to_datetime(
                    df["결산년월"].astype(str), format="%Y%m"
                )

                return df
            else:
                return pd.DataFrame()
        else:
            print(f"Error: {response.status_code}")
            return pd.DataFrame()
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return pd.DataFrame()


def get_stock_chart(access_token, appkey, appsecret, params, is_mock=True):
    """
    주가 차트 데이터를 가져오는 함수
    """
    if is_mock:
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    else:
        url = "https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": "FHKST03010100",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()

            if "output2" in data and data["output2"]:
                df = pd.DataFrame(data["output2"])

                # 컬럼명을 한글로 변경
                column_mapping = {
                    "stck_bsop_date": "날짜",
                    "stck_oprc": "시가",
                    "stck_hgpr": "고가",
                    "stck_lwpr": "저가",
                    "stck_clpr": "종가",
                    "acml_vol": "거래량",
                    "acml_tr_pbmn": "거래대금",
                    "prdy_vrss": "전일대비",
                    "prdy_vrss_sign": "전일대비구분",
                }
                df = df.rename(columns=column_mapping)

                # 데이터 타입 변환
                numeric_columns = [
                    "시가",
                    "고가",
                    "저가",
                    "종가",
                    "거래량",
                    "거래대금",
                    "전일대비",
                ]
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # 날짜 형식 변환 (YYYYMMDD -> datetime)
                df["날짜"] = pd.to_datetime(df["날짜"], format="%Y%m%d")

                # 전일대비구분 변환
                sign_mapping = {
                    "1": "상한가",
                    "2": "상승",
                    "3": "보합",
                    "4": "하한가",
                    "5": "하락",
                }
                df["전일대비구분"] = df["전일대비구분"].map(sign_mapping)

                # 날짜 기준으로 정렬
                df = df.sort_values(by="날짜")

                # 거래대금 단위를 백만원으로 변환
                df["거래대금"] = df["거래대금"] / 1_000_000

                return df
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except requests.RequestException as e:
        return pd.DataFrame()


def add_indicators(df):
    """기술적 지표 계산"""
    # 이동평균선
    df["MA5"] = df["종가"].rolling(5).mean()
    df["MA20"] = df["종가"].rolling(20).mean()
    df["MA60"] = df["종가"].rolling(60).mean()

    # RSI
    delta = df["종가"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD
    exp12 = df["종가"].ewm(span=12, adjust=False).mean()
    exp26 = df["종가"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드
    df["BB_middle"] = df["종가"].rolling(window=20).mean()
    df["BB_upper"] = df["BB_middle"] + 2 * df["종가"].rolling(window=20).std()
    df["BB_lower"] = df["BB_middle"] - 2 * df["종가"].rolling(window=20).std()

    # 거래량 지표
    df["Volume_MA5"] = df["거래량"].rolling(5).mean()
    df["Volume_MA20"] = df["거래량"].rolling(20).mean()
    df["Volume_Ratio"] = df["거래량"] / df["Volume_MA20"]

    # 가격 변동성
    df["Price_Volatility"] = df["종가"].rolling(window=20).std()

    return df.ffill().bfill()


def fetch_stock_data_5_years(
    access_token, appkey, appsecret, stock_code, start_date, end_date, is_mock=True
):
    """
    5년 치 데이터를 100일 단위로 조회하는 함수.

    Parameters:
    - access_token: API 접근 토큰
    - appkey: 앱키
    - appsecret: 앱시크릿
    - stock_code: 종목코드
    - start_date: 시작 날짜 (YYYYMMDD 형식)
    - end_date: 종료 날짜 (YYYYMMDD 형식)
    - is_mock: 모의 환경 여부

    Returns:
    - DataFrame: 조회된 주가 데이터를 통합한 DataFrame
    """
    current_start_date = datetime.strptime(start_date, "%Y%m%d")
    current_end_date = current_start_date + timedelta(days=99)

    all_data = []

    while current_start_date <= datetime.strptime(end_date, "%Y%m%d"):
        # 종료일이 end_date를 초과하지 않도록 조정
        if current_end_date > datetime.strptime(end_date, "%Y%m%d"):
            current_end_date = datetime.strptime(end_date, "%Y%m%d")

        # 파라미터 설정
        query_params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_DATE_1": current_start_date.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": current_end_date.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }

        # 주가 데이터 호출
        data_chunk = get_stock_chart(
            access_token, appkey, appsecret, query_params, is_mock
        )
        if not data_chunk.empty:
            all_data.append(data_chunk)

        # 날짜 이동 (다음 100일 구간)
        current_start_date += timedelta(days=100)
        current_end_date += timedelta(days=100)

    # 결과 데이터프레임 생성
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # 중복 제거(혹시 기간이 겹치는 경우)
        combined_df.drop_duplicates(subset=["날짜"], keep="last", inplace=True)
        # 날짜 기준으로 정렬
        combined_df.sort_values(by="날짜", inplace=True)
        return combined_df
    else:
        return pd.DataFrame()


# 메인 실행
if __name__ == "__main__":
    # 환경 변수에서 앱키와 앱시크릿 가져오기
    appkey = os.getenv("KOREA_INVESTMENT_APPKEY")
    appsecret = os.getenv("KOREA_INVESTMENT_APPSECRET")

    if not appkey or not appsecret:
        print("Error: 환경 변수에서 앱키 또는 앱시크릿을 찾을 수 없습니다.")
        exit(1)

    # 접근 토큰 가져오기 (캐싱 또는 새 발급)
    access_token = get_access_token(appkey, appsecret, is_mock=False)  # 실전 환경

    if access_token:
        # 예시: 5년치 데이터 조회
        stock_code = "005930"  # 삼성전자
        start_date = "20190101"  # 5년 전(2019-01-01)
        end_date = "20231231"  # 현재 혹은 원하는 종료일

        # 1. 5년 치 주가 데이터 조회
        stock_data_5_years = fetch_stock_data_5_years(
            access_token,
            appkey,
            appsecret,
            stock_code,
            start_date,
            end_date,
            is_mock=False,
        )
        print("\n[5년 치 주가 데이터]")
        print(stock_data_5_years)

        # 2. 재무 데이터 가져오기 (연간 데이터 기준)
        financial_data = get_financial_ratio(
            access_token, appkey, appsecret, stock_code, div_code="0"
        )
        print("\n[재무비율 데이터]")
        print(financial_data)

        # 3. 재무 데이터 병합
        if not stock_data_5_years.empty and not financial_data.empty:
            # (1) 결산년월을 해당 월 마지막 날짜로 변경
            financial_data["결산년월"] = financial_data[
                "결산년월"
            ] + pd.offsets.MonthEnd(0)

            # (2) merge_asof
            stock_data_5_years = stock_data_5_years.sort_values("날짜")
            financial_data = financial_data.sort_values("결산년월")

            merged_data = pd.merge_asof(
                stock_data_5_years,
                financial_data,
                left_on="날짜",
                right_on="결산년월",
                direction="backward",  # 주가 날짜보다 이전의 재무 데이터를 매칭
            )

            merged_data = add_indicators(merged_data)

            print("\n[병합된 데이터]")
            print(merged_data)

            # 필요하다면 CSV 저장
            cleaned_data = merged_data.dropna(axis=1, how="any")
            cleaned_data.to_csv(
                "merged_5_years_data_cleaned.csv", index=False, encoding="utf-8-sig"
            )
        else:
            print("재무 데이터 또는 주가 데이터가 비어있어 병합할 수 없습니다.")

    else:
        print("Error: 접근 토큰 가져오기 실패")
