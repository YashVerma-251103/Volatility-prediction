from imports import *
# # from DataPipeline import DataPipeline
# from ModelBuilder import ModelBuilder
# from VolatilityPredictor import VolatilityPredictor
# from VolatilityPredictionSystem import VolatilityPredictionSystem


class DataPipeline:

    # Constructor for pipeline.
    def __init__(self, asset, trading_day_window=30, active_total_trading_days=252):
        self.asset = asset
        self.trading_window = trading_day_window
        self.active_days = active_total_trading_days
        self.scaling_factor = np.sqrt(active_total_trading_days)
        self.dataframe = None
        self.features = None
        self.scaled_features = None

    ## Fetching past for the asset for this interval of time.
    def fetch_financial_data(self, start_date, end_date):

        # this dataframe holds OHLCV data. (Open, HIgh, Low, Close, Volume).
        # dataframe = yf.download(tickers=self.asset, start=start_date, end=end_date)
        dataframe = yf.download(self.asset, start=start_date, end=end_date)

        ## Calculating returns.
        dataframe["Returns"] = np.log(dataframe["Close"] / dataframe["Close"].shift(1))

        ## Returning the dataframe with returns calculated.
        self.dataframe = dataframe
        return dataframe

    ## Calculating Realized Volatility -> Standard Deviation of Returns from the Mean Return.
    def calculate_realized_volatility(self):
        # taken a month by default can change it accordingly.

        ## Calculating the Annual Volatility using the Rolling Window Standard Deviation and Scaling it.
        self.dataframe["RealizedVolatility"] = (
            self.dataframe["Returns"].rolling(window=self.trading_window).std()
        ) * (self.scaling_factor)

        ## Calculating High-Low Volatility
        self.dataframe["HighLowVolatility"] = np.log(
            self.dataframe["High"] / self.dataframe["Low"]
        )

        ## Calculating GARMAN KLASS Volatility using the mathematical formula
        self.dataframe["GarmanKlassVolatility"] = np.sqrt(
            (((np.log(self.dataframe["High"] / self.dataframe["Low"])) ** 2) * 0.5)
            - (
                ((2 * (np.log(2))) - 1)
                * ((np.log(self.dataframe["Close"] / self.dataframe["Open"])) ** 2)
            )
        )

        return self.dataframe

    ## Technical Indicators and features for Volatility Prediction
    def create_feature_set(self, Moving_AvgStd_window=22, lookback_period=[5, 10, 22]):

        # Ensuring the returns are present
        if "Returns" not in self.dataframe.columns:
            self.dataframe["Returns"] = np.log(
                self.dataframe["Close"] / self.dataframe["Close"].shift(1)
            )

        ## Calculating features
        # if self.features is None:
        #     self.features = pd.DataFrame(index=self.dataframe.index)
        self.features = pd.DataFrame(index=self.dataframe.index)

        ### Volume-based features
        #### this is the rolling mean
        self.features["Volume_MovAvg"] = (
            self.dataframe["Volume"].rolling(window=Moving_AvgStd_window).mean()
        )
        #### this is the rolling standard deviation
        self.features["Volume_StdDev"] = (
            self.dataframe["Volume"].rolling(window=Moving_AvgStd_window).std()
        )

        ### Price-based features
        for period in lookback_period:

            #### Moving avgs
            self.features[f"Price_MovAvg_{period}"] = (
                self.dataframe["Close"].rolling(window=period).mean()
            )
            self.features[f"Price_StdDev_{period}"] = (
                self.dataframe["Close"].rolling(window=period).std()
            )

            #### RSI -> Relative Strength Index ==> OverBought or OverSold
            delta = self.dataframe["Close"].diff()
            avg_gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            ##### if the where condition is not satisfied then we replace it by 0
            avg_loss = (-(delta.where(delta < 0, 0))).rolling(window=period).mean()
            relative_strength = avg_gain / avg_loss
            self.features[f"RSI_{period}"] = 100 - (100 / (1 + relative_strength))

            #### Historical Volatility
            self.features[f"Hist_Vol_{period}"] = (
                self.dataframe["Returns"].rolling(window=period).std()
            ) * self.scaling_factor

        ### VWAP-> Volume Weighted Average Price
        self.features["VWAP"] = (((
            self.dataframe["Close"] * self.dataframe["Volume"]
        ).cumsum()) / ((self.dataframe["Volume"]).cumsum()))

        return self.features

    ## Preparing complete dataset for the training volatility predictor model
    def prepare_training_data(
        self,
        start_date,
        end_date,
        prediction_horizon=5,
        Moving_AvgStd_window=22,
        lookback_period=[5, 10, 22]
    ):
        self.fetch_financial_data(start_date, end_date)
        self.calculate_realized_volatility()
        self.create_feature_set(Moving_AvgStd_window, lookback_period)

        # Target variable -> Future Volatility
        target = self.dataframe["RealizedVolatility"].shift(-(prediction_horizon))

        # Combining features and target
        final_dataset = pd.concat(
            [self.features, target], axis=1
        ).dropna()  #### dropping the columns with missing values from the data set

        # Getting the feature names before scaling
        feature_cols = final_dataset.columns[:-1]  ### all columns except the target
        print(feature_cols.shape)

        ### starting the standard scaler
        scaler = ssc()

        print(final_dataset.iloc[:,:-1].shape)

        # Fit and transform the features (target not included)
        scaled_features = scaler.fit_transform(final_dataset.iloc[:, :-1])

        ## Convert scaled features back to dataframe with correct column names
        self.scaled_features_df = pd.DataFrame(
            scaled_features, index=final_dataset.index, columns=feature_cols
        )

        return self.scaled_features_df, final_dataset["RealizedVolatility"], scaler