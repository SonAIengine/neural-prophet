import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralprophet import NeuralProphet


class StockPredictor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df["날짜"] = pd.to_datetime(self.df["날짜"])
        self.scaler = StandardScaler()

    def prepare_neuralprophet_data(self):
        prophet_df = self.df[
            ["날짜", "종가", "MA5", "MA20", "MA60", "RSI", "MACD"]
        ].copy()
        prophet_df.columns = ["ds", "y", "ma5", "ma20", "ma60", "rsi", "macd"]
        return prophet_df.ffill().bfill()

    def train_and_predict(self, train_data, days=30):
        model = NeuralProphet(
            n_forecasts=days,
            n_lags=30,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            learning_rate=0.01,
            epochs=100,
        )

        # Lagged Regressors 등록
        model = model.add_lagged_regressor(names=["ma5", "ma20", "ma60", "rsi", "macd"])

        # 데이터 학습
        metrics = model.fit(train_data, freq="D")

        # 미래 예측
        future = model.make_future_dataframe(
            train_data, periods=days, n_historic_predictions=len(train_data)
        )
        forecast = model.predict(future)

        return model, forecast

    def evaluate(self, test_df, forecast_df):
        test_df = test_df.set_index("ds")
        forecast_df = forecast_df.set_index("ds")

        if "yhat30" not in forecast_df.columns:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

        # 공통 날짜 확인
        common_dates = test_df.index.intersection(forecast_df.index)
        if len(common_dates) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

        actual = test_df.loc[common_dates, "y"]
        predicted = forecast_df.loc[common_dates, "yhat30"]  # 30일 예측값 기준

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

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

        # 예측 데이터
        if "yhat30" in forecast.columns:
            plt.plot(
                forecast["ds"],
                forecast["yhat30"],
                label="Predicted (30-day)",
                alpha=0.8,
                linewidth=2,
                linestyle="--",
                color="green",
            )

        plt.title(
            "Stock Price Prediction (NeuralProphet 30 Days Forecast)",
            fontsize=14,
            pad=20,
        )
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

        predictor = StockPredictor("merged_5_years_data_cleaned.csv")
        prophet_df = predictor.prepare_neuralprophet_data()

        # 데이터 분할 (최근 30일을 테스트 세트로 사용)
        train_size = len(prophet_df) - 30
        train_df = prophet_df.iloc[:train_size].copy()
        test_df = prophet_df.iloc[train_size:].copy()

        # 예측
        model, forecast = predictor.train_and_predict(train_df, days=30)

        # 시각화
        save_path = os.path.join(current_dir, "stock_prediction_neuralprophet_30days.png")
        predictor.plot(model, forecast, train_df, test_df, save_path)

        # 성능 평가
        metrics = predictor.evaluate(test_df, forecast)
        print("\n예측 성능:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        # 미래 예측 출력
        if "yhat30" in forecast.columns:
            future_30 = forecast[["ds", "yhat30"]]
            future_30.columns = ["날짜", "예측값(30일)"]
            print("\n다음 30일 예측(yhat30):")
            print(future_30.tail(30).to_string(index=False))
        else:
            print("\n예측 결과가 충분하지 않습니다. yhat30을 찾을 수 없습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()
