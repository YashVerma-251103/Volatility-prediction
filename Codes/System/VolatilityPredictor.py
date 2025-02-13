from imports import *
# from DataPipeline import DataPipeline
# from ModelBuilder import ModelBuilder
# # from VolatilityPredictor import VolatilityPredictor
# from VolatilityPredictionSystem import VolatilityPredictionSystem



class VolatilityPredictor:

    # Constructor
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.model = None

    # Create XGBoost model with optimized parameters for volatility prediction
    def create_xgboost_model(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    ):
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            random_state=random_state,
        )

    # Create LSTM model for sequence-based volatility prediction
    def create_lstm_model(
        self,
        input_shape,
        no_lstm_layers=50,
        dropout=0.2,
        dense=1,
        optimizer="adam",
        loss="mse"
    ):
        model = seqMD(
            [
                LSTM(no_lstm_layers, return_sequences=True, input_shape=input_shape),
                dpMD(dropout),
                LSTM(no_lstm_layers),
                dpMD(dropout),
                Dense(dense)
            ]
        )
        model.compile(optimizer=optimizer, loss=loss)
        return model

    # Prepare for LSTM model
    def prepare_LSTM_sequences(self, X, sequence_length=10):
        sequences = []
        lenX = len(X)
        for i in range(lenX - sequence_length):
            sequences.append(X[i : (i + sequence_length)])
        return np.array(sequences)

    # Training on Volatility prediction model
    def train(
        self, X, y, validation_split=0.2, target_seq_len=10, epochs=50, batch_size=32, create_new_model=False
    ):
        if self.model_type == "xgboost":
            if (create_new_model):
                self.model = self.create_xgboost_model()
            self.model.fit(X, y)
            return
        elif self.model_type == "LSTM":
            X_seq = self.prepare_LSTM_sequences(X, target_seq_len)
            y_seq = y[target_seq_len:] # adjust for target for the sequence length
            if (create_new_model):
                self.model=self.create_lstm_model((X_seq.shape[1],X_seq.shape[2]))
            self.model.fit(
                X_seq,
                y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )
            return
        

    # Evaluation of model based on performance using multiple metrics
    def evaluate(self, X_test, y_test,target_seq_len=10):
        if self.model_type == "xgboost":
            prediction = self.model.predict(X_test)
        elif self.model_type == "LSTM":
            X_test_seq = self.prepare_LSTM_sequences(X_test)
            prediction = self.model.predict(X_test_seq)
            y_test=y_test[target_seq_len:]

        metrics = {
            "RMSE": np.sqrt(mse(y_test, prediction)),  ## root mean square error
            "MAE": mae(y_test, prediction),  ## mean absolute error
            "R2": r2s(y_test, prediction),  ## r2 score
        }

        return metrics,prediction

    # Making the volatility predictions on new data
    def predict(self, X):
        if self.model_type == "xgboost":
            return self.model.predict(X)
        elif self.model_type == "LSTM":
            return self.model.predict(self.prepare_LSTM_sequences(X))