import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


class StockPredictor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df["날짜"] = pd.to_datetime(self.df["날짜"])
        self.scaler = StandardScaler()

    def add_indicators(self):
        """기술적 지표 계산"""
        df = self.df.copy()

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

        self.df = df.ffill().bfill()
        return self

    def prepare_prophet_data(self):
        """Prophet 모델용 데이터 준비"""
        # 최근 데이터 중심으로 스케일링
        recent_window = min(60, len(self.df))  # 최근 60일 또는 전체 데이터
        price_scaler = StandardScaler().fit(
            self.df[
                ["종가", "MA5", "MA20", "MA60", "BB_middle", "BB_upper", "BB_lower"]
            ].tail(recent_window)
        )
        volume_scaler = StandardScaler().fit(
            self.df[["거래량", "Volume_MA5", "Volume_MA20"]].tail(recent_window)
        )

        scaled_price = price_scaler.transform(
            self.df[
                ["종가", "MA5", "MA20", "MA60", "BB_middle", "BB_upper", "BB_lower"]
            ]
        )
        scaled_volume = volume_scaler.transform(
            self.df[["거래량", "Volume_MA5", "Volume_MA20"]]
        )

        prophet_df = pd.DataFrame(
            {
                "ds": self.df["날짜"],
                "y": self.df["종가"],
                "ma5": scaled_price[:, 1],
                "ma20": scaled_price[:, 2],
                "ma60": scaled_price[:, 3],
                "bb_middle": scaled_price[:, 4],
                "bb_upper": scaled_price[:, 5],
                "bb_lower": scaled_price[:, 6],
                "volume": scaled_volume[:, 0],
                "volume_ma5": scaled_volume[:, 1],
                "volume_ma20": scaled_volume[:, 2],
                "volume_ratio": self.df["Volume_Ratio"],
                "rsi": self.df["RSI"] / 100,
                "macd": self.df["MACD"],
                "volatility": self.df["Price_Volatility"] / self.df["종가"].mean(),
            }
        )

        return prophet_df.ffill().bfill()

    def train_and_predict(self, train_data, days=30):
        """모델 학습 및 예측"""
        # 최근 데이터에 더 큰 가중치 부여
        train_data = train_data.copy()
        dates = pd.to_datetime(train_data["ds"])
        max_date = dates.max()
        train_data["weights"] = [
            (1 + 0.1 * (d - max_date).days / 365) ** 2 for d in dates
        ]

        model = Prophet(
            changepoint_prior_scale=0.05,  # 기본값: 변화점 민감도
            seasonality_prior_scale=10.0,  # 기본값: 계절성 영향
            holidays_prior_scale=10.0,  # 기본값: 휴일 효과
            daily_seasonality=True,  # 기본값: 일일 계절성
            weekly_seasonality=True,  # 기본값: 주간 계절성
            yearly_seasonality=True,  # 기본값: 연간 계절성
            interval_width=0.8,  # 기본값: 예측 구간
            changepoint_range=0.8,  # 기본값: 변화점 허용 범위
            n_changepoints=25  # 기본값: 변화점 수
        )

        regressors = [
            "ma5",
            "ma20",
            "ma60",
            "rsi",
            "macd",
        ]

        for col in regressors:
            if col in train_data.columns:
                model.add_regressor(col, mode="multiplicative", standardize=False)

        model.fit(train_data)

        future = model.make_future_dataframe(periods=days)

        # 미래 데이터에 대한 특성값 설정 (이동평균으로 추정)
        last_date = train_data["ds"].max()
        for col in regressors:
            if col in train_data.columns:
                future.loc[future["ds"] > last_date, col] = (
                    train_data[col].rolling(window=30).mean().iloc[-1]
                )
                future.loc[future["ds"] <= last_date, col] = train_data[col]

        forecast = model.predict(future)
        return model, forecast

    def evaluate(self, test_df, forecast_df):
        """예측 성능 평가"""
        test_df = test_df.set_index("ds")
        forecast_df = forecast_df.set_index("ds")

        common_dates = test_df.index.intersection(forecast_df.index)

        if len(common_dates) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

        actual = test_df.loc[common_dates, "y"]
        predicted = forecast_df.loc[common_dates, "yhat"]

        metrics = {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAPE": np.mean(np.abs((actual - predicted) / actual)) * 100,
        }
        return metrics

    def plot(self, model, forecast, train_df, test_df, save_path="prediction.png"):
        """결과 시각화"""
        save_path = os.path.abspath(save_path)

        plt.figure(figsize=(15, 8))
        plt.clf()

        # 학습 데이터
        plt.plot(
            train_df["ds"], train_df["y"], label="Training", alpha=0.8, linewidth=1
        )

        # 테스트 데이터
        plt.plot(test_df["ds"], test_df["y"], label="Test", alpha=0.8, linewidth=1)

        # 예측값 (전체 기간)
        plt.plot(
            forecast["ds"],
            forecast["yhat"],
            label="Predicted",
            alpha=0.8,
            linewidth=2,
            linestyle="--",
            color="green",
        )

        # 예측 구간
        plt.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            alpha=0.2,
            label="Prediction Interval",
            color="gray",
        )

        # 미래 30일 예측 구간 강조
        future_dates = forecast["ds"] > train_df["ds"].max()
        plt.fill_between(
            forecast.loc[future_dates, "ds"],
            forecast.loc[future_dates, "yhat_lower"],
            forecast.loc[future_dates, "yhat_upper"],
            alpha=0.3,
            color="lightblue",
            label="Future Prediction",
        )

        plt.title("Stock Price Prediction (30 Days Forecast)", fontsize=14, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend(fontsize=10, loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"그래프가 저장되었습니다: {save_path}")
        except Exception as e:
            print(f"그래프 저장 중 오류 발생: {str(e)}")
        finally:
            plt.close()


def main():
    try:
        current_dir = os.getcwd()
        print(f"현재 작업 디렉토리: {current_dir}")

        predictor = StockPredictor("merged_5_years_data.csv")
        predictor.add_indicators()
        prophet_df = predictor.prepare_prophet_data()

        # 데이터 분할 (최근 30일을 테스트 세트로 사용)
        train_size = len(prophet_df) - 30
        train_df = prophet_df.iloc[:train_size].copy()
        test_df = prophet_df.iloc[train_size:].copy()

        # 예측 (30일)
        model, forecast = predictor.train_and_predict(train_df, days=30)

        # 시각화
        save_path = os.path.join(current_dir, "stock_prediction_30days.png")
        predictor.plot(model, forecast, train_df, test_df, save_path)

        # 성능 평가
        metrics = predictor.evaluate(test_df, forecast)

        print("\n예측 성능:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        print("\n다음 30일 예측:")
        future = forecast.tail(30)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        future.columns = ["날짜", "예측값", "하한", "상한"]
        print(future.to_string(index=False))

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()
