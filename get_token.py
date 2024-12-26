import requests


def get_access_token(appkey, appsecret, is_mock=True):
    # API URL 설정
    if is_mock:
        url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"  # 모의 환경
    else:
        url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"  # 실전 환경

    # 요청 본문 데이터
    payload = {
        "grant_type": "client_credentials",
        "appkey": appkey,
        "appsecret": appsecret,
    }

    # HTTP 헤더 설정
    headers = {"Content-Type": "application/json; charset=UTF-8"}

    try:
        # POST 요청
        response = requests.post(url, json=payload, headers=headers)

        # 응답 처리
        if response.status_code == 200:
            return response.json()  # 성공 시 응답 JSON 반환
        else:
            # 에러 발생 시 상태 코드와 메시지 반환
            return {"error": response.status_code, "message": response.text}
    except requests.RequestException as e:
        # 요청 중 예외 발생 처리
        return {"error": "RequestException", "message": str(e)}
