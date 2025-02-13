# please insatll the following libraries.
# numpy pandas pandas_datareader matplotlib yfinance scikit-learn xgboost tensorflow mlflow


# stat data analyzing libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as pdata
from datetime import datetime as dtim, timedelta as tdel

# projecting data libraries
import matplotlib.pyplot as plt

# financial data libraries
import yfinance as yf

# ML libraries
import mlflow as mlf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler as ssc
from sklearn.model_selection import TimeSeriesSplit as tss
from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    r2_score as r2s,
)
from tensorflow.keras.models import Sequential as seqMD
from tensorflow.keras.layers import LSTM, Dense, Dropout as dpMD

# other miscellaneous libraries
import json
import joblib
