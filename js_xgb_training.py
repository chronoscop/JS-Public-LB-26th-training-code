
import pandas as pd
import numpy as np
import os 
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
import gc
from sklearn.metrics import r2_score
import xgboost as xgb
from tqdm.auto import tqdm
import joblib


class CONFIG:
    target_col = "responder_6"
    feature_cols = ["symbol_id", "time_id"] \
        + [f"feature_{idx:02d}" for idx in range(79)] \
        + [f"responder_{idx}_lag_1" for idx in range(9)]
    categorical_cols = []

def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:  # 遍历每列的列名
        col_type = df[col].dtype  # 列名的类型
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df.loc[:, col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df.loc[:, col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df.loc[:, col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df.loc[:, col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df.loc[:, col] = df[col].astype(np.float32)
                    else:
                        df.loc[:, col] = df[col].astype(np.float16)  
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df.loc[:, col] = df[col].astype(np.float32)
                else:
                    df.loc[:, col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

    
file_paths = [f"/kaggle/input/janestree-process-data/train_folder/partition_id={i}/train_data_0.parquet" for i in range(5, 9)]
file_paths.append("/kaggle/input/janestree-process-data/val_folder/partition_id=1/val_data_0.parquet")
file_paths.append("/kaggle/input/janestree-process-data/val_folder/partition_id=2/val_data_0.parquet")
dfs = [pl.read_parquet(file) for file in file_paths]
train_df = pl.concat(dfs)
del file_paths, dfs
gc.collect()
print("Final shape:", train_df.shape)



df_lazy = train_df.lazy()

fill_exprs = [pl.col(c).fill_null(0) for c in train_df.columns]
df_lazy = df_lazy.with_columns(*fill_exprs)
train_df = df_lazy.collect()  # 在最后收集计算结果
del df_lazy, fill_exprs
gc.collect()
    
print("Total null values:", train_df.null_count().select(pl.all().sum()).to_series()[0])


train=train_df
del train_df
gc.collect()


train=train.to_pandas()
train.dtypes.value_counts()

X_train = train[CONFIG.feature_cols].copy()
y_train = train[CONFIG.target_col].copy()
w_train = train["weight"].copy()
del train
gc.collect()

# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        numerator = np.sum((y_pred - y_true) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        numerator = np.sum(sample_weight * (y_pred - y_true) ** 2)
        denominator = np.sum(sample_weight * (y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (numerator / denominator)
    return 'r2', r2


# 定义 XGBoost 参数
params = {
    'objective': 'reg:squarederror',  
    'random_state': 1212,
    'tree_method': 'gpu_hist',
    'device' : 'cuda',
    'learning_rate': 0.02156022412857549, 
    'max_depth': 8, 
    'subsample': 0.7697954003310141, 
    'colsample_bytree': 0.5182134365961873, 
    'reg_alpha': 0.0032315937370696354, 
    'reg_lambda': 0.002663721647776419
}

# 将训练数据和验证数据转换为 DMatrix 格式
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train) 
del X_train, y_train, w_train, 

model_xgb = xgb.train(
    params, 
    dtrain, 
    num_boost_round=471,  
    verbose_eval=25  
)
del dtrain
gc.collect()

# os.system('mkdir models')
joblib.dump(model_xgb, '../models/xgb_LB.model')
print("finish training")