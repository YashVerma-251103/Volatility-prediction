from imports import *
from DataPipeline import DataPipeline
# # from ModelBuilder import ModelBuilder
from VolatilityPredictor import VolatilityPredictor
# from VolatilityPredictionSystem import VolatilityPredictionSystem


class ModelBuilder:
    def __init__(
        self,
        asset,
        input_shape,
        start_date,
        end_date,
        test_start,
        test_end,
        pipeline=None,
        trading_day_window=30,
        active_total_trading_days=252,
        model_type="xgboost",
        no_lstm_layers=50,
        dropout=0.2,
        dense=1,
        optimizer="adam",
        loss="mse",
    ):
        assert trading_day_window > 0 and active_total_trading_days > trading_day_window
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.test_start = test_start
        self.test_end = test_end
        if (pipeline==None):
            self.pipeline = DataPipeline(
                asset=asset,
                trading_day_window=trading_day_window,
                active_total_trading_days=active_total_trading_days,
            )
        else:
            assert pipeline.__class__ == DataPipeline
            self.pipeline=pipeline
        if model_type == "xgboost":
            self.predictor = VolatilityPredictor(model_type=model_type)
            self.predictor.model = self.predictor.create_xgboost_model()
        elif model_type == "LSTM":
            assert input_shape != None
            self.predictor = VolatilityPredictor(model_type=model_type)
            self.predictor.model = self.predictor.create_lstm_model(
                input_shape, no_lstm_layers, dropout, dense, optimizer, loss
            )
        return

    def model_train(self, train_start, train_end, test_start, test_end):
        print(f"Preparing Training Data from Yahoo Finance for Asset=> {self.asset}")
        X_train, y_train, Train_scaler = self.pipeline.prepare_training_data(
            train_start, train_end
        )

        print("Training Model")
        self.predictor.train(X_train, y_train)

        print("Preparing Test Data")
        X_test, y_test, Test_scaler = self.pipeline.prepare_training_data(
            test_start, test_end
        )

        print("Making Prediction")
        metrics, predictions = self.predictor.evaluate(X_test, y_test)

        print("Calculating the additional Metrics")
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        print("Preparing Results")
        results = {
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "MAPE": mape,
            "Avg. Predicted Volatility": np.mean(predictions),
            "Avg. Actual Volatility": np.mean(y_test),
            "Prediction Bias": np.mean(predictions - y_test),
            "Max Prediction Error": np.max(np.abs(predictions - y_test)),
        }

        print("Creating Comparision Plot")
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test.values, label="Actual Volatility")
        plt.plot(y_test.index, predictions, label="Predicted Volatility")
        plt.title(f"{self.asset} Volatility - Actual V/s Predicted")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)
        plt.show()

        return results, predictions, y_test

    def run_backtest(self):
        # print("Starting Backtests")
        result, predictions, actuals = self.model_train(
            self.start_date, self.train_end, self.test_start, self.test_end
        )

        # print("\n\nBackTest Results:")
        # for metric, value in result.items():
        #     print(f"{metric}:{value:.4f}")

        # print("\nCalculating Prediction Accuracy Bands")
        within_5_percent = np.mean(np.abs((predictions-actuals)/actuals) <= 0.05)*100
        within_10_percent = np.mean(np.abs((predictions-actuals)/actuals) <= 0.10)*100

        # print("\n\nPrediction Accuracy:")
        # print(f"Prediction within 5% of actual: {within_5_percent:.2f}%")
        # print(f"Prediction within 10% of actual: {within_10_percent:.2f}%")