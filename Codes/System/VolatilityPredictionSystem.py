# from imports import *
# from DataPipeline import DataPipeline
# from ModelBuilder import ModelBuilder
# from VolatilityPredictor import VolatilityPredictor
# # from VolatilityPredictionSystem import VolatilityPredictionSystem



class VolatilityPredictionSystem:
    def __init__(self, model_path=None):

        # assert model_path != None

        self.model = None
        self.scaler = None
        self.performance_log = []
        if model_path is not None:
            self.load_model(model_path)

    ## loading the trained model and scaler from disk/memory
    def load_model(self, model_path):
        assert model_path is not None

        self.model = joblib.load(f"{model_path}/model.joblib")
        self.scaler = joblib.load(f"{model_path}/scaler.joblib")

    ## save the trained model and scaler to disk/memory
    def save_model(self, model_path):
        assert model_path is not None

        joblib.dump(self.model, f"{model_path}/model.joblib")
        joblib.dump(self.scaler, f"{model_path}/scaler.joblib")

    ## Evaluate different prediction accuracy over different time horizons
    def evaluate_predictions(self, predictions, actual_values, evaluation_window=30):
        assert predictions is not None
        assert actual_values is not None
        assert evaluation_window is not None
        ### by default evaluation window is of one month

        ### calculating the error metrics
        rmse = np.sqrt(mse(actual_values, predictions))

        ### calculating the directional accuracy
        directional_correct = np.sum(
            np.sign(predictions[1:] - predictions[:-1])
            == np.sign(actual_values[1:] - actual_values[:-1])
        )

        directional_accuracy = directional_correct / (len(predictions) - 1)

        ## Calculating Running Volatility prediction error
        running_error = (
            pd.Series(predictions - actual_values).rolling(evaluation_window).std()
        )

        evaluation = {
            "rmse": rmse,
            "directional_accuracy": directional_accuracy,
            "average_running_error": running_error.mean(),
            "max_running_error": running_error.max(),
        }

        self.performance_log.append(
            {
                "time_stamp": dtim.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": evaluation,
            }
        )

        return evaluation

    ## Generate Prediction report
    def generate_prediction_report(
        self, asset, predictions, actual_values, start_date, end_date
    ):
        assert asset is not None
        assert predictions is not None
        assert actual_values is not None
        assert isinstance(start_date, dtim)
        assert isinstance(end_date, dtim)

        report = {
            "asset": asset,
            "prediction_period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            },
            "evaluation_metrics": self.evaluate_predictions(predictions, actual_values),
            "prediction_summary": {
                "mean_predicted_volatility": float(np.mean(predictions)),
                "max_predicted_volatility": float(np.max(predictions)),
                "min_predicted_volatility": float(np.min(predictions)),
                "volatility_trend": (
                    "Increasing" if predictions[-1] > predictions[0] else "Decreasing"
                ),
            },
        }

        return report

    ## Monitor model Performance overtime to detect drift.
    def monitor_model_drift(
        self, prediction_window=30, accuracy_degrading_threshold=0.6
    ):
        assert isinstance(prediction_window, int) and (prediction_window >= 1)
        assert isinstance(accuracy_degrading_threshold, float) and (
            0 < accuracy_degrading_threshold < 1
        )

        if len(self.performance_log) < 2:
            return None

        recent_performance = pd.DataFrame(
            [log["metrics"] for log in self.performance_log[-prediction_window:]]
        )

        drift_analysis = {
            "rmse_trend": recent_performance["rmse"].is_monotonic_increasing,
            "accuracy_degradation": recent_performance["directional_accuracy"].mean()
            < accuracy_degrading_threshold,
            "error_volatility": recent_performance["average_running_error"].std(),
            "requires_restraining": False,
        }

        ### setting the restraining flag if performance is degrading
        if drift_analysis["rmse_trend"] and drift_analysis["accuracy_degradation"]:
            drift_analysis["requires_restraining"] = True

        return drift_analysis
    
    ## Deploy model for real time predictions
    def deploy_model(self,asset,prediction_horizon=5,recent_data_days = 100):
        assert asset is not None
        assert (isinstance(prediction_horizon,int) and (prediction_horizon>0))
        assert (isinstance(recent_data_days,int) and (recent_data_days>0))


        today = dtim.now()
        start_date = today - tdel(days=recent_data_days)


        ### Fetch latest data
        Dpipe = DataPipeline(asset=asset)
        data = Dpipe.fetch_financial_data(start_date=start_date,end_date=today)
        features = Dpipe.create_feature_set()

        ### Scaling features
        scaled_features = self.scaler.transform(features.iloc[-prediction_horizon:])

        ### Make Prediction
        prediction = self.model.predict(scaled_features)

        return {
            "asset" : asset,
            "prediction_date":today.strftime('%Y-%m-%d'),
            "predicted_volatility":float(prediction[-1]),
            "confidence_metrics":self.monitor_model_drift()
        }