from imports import *
from DataPipeline import DataPipeline
from ModelBuilder import ModelBuilder
from VolatilityPredictor import VolatilityPredictor
from VolatilityPredictionSystem import VolatilityPredictionSystem


asset = "AAPL"
train_start = "2020-01-01"
train_end = "2023-01-01"
test_start = "2023-01-02"
test_end = "2024-01-01"

sdp = DataPipeline(asset)
# svp = VolatilityPredictor("xgboost")
smd = ModelBuilder(asset=asset,input_shape=None,start_date=train_start,end_date=train_end,test_start=test_start,test_end=test_end,pipeline=sdp)
smd.model_train(smd.start_date,smd.end_date,smd.test_start,smd.test_end)