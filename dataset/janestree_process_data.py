import pandas as pd
import polars as pl
import numpy as np
import gc
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedGroupKFold

target_col = "responder_6"
lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}

# change to your data path
train = pl.scan_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet")
train_df = train.collect()

# 获取 DataFrame 的行数和列数
print(train_df.shape[0])  # 行数
print(train_df.shape[1])  # 列数
print(train_df["partition_id"])
del train_df
gc.collect()

lags=train.select(pl.col(lag_cols_original))
lags=lags.rename(lag_cols_rename)
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lagged by 1 day
    )
lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  # pick up last record of previous date
lags.show_graph()

train = train.join(lags, on=["date_id", "symbol_id"],  how="left")
train.show_graph()

len_train = train.select(pl.col("date_id")).collect().shape[0]
valid_records = int(len_train * 0.2)
len_ofl_mdl = len_train - valid_records
last_tr_dt  = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

print(f"\n len_train = {len_train}")
print(f"\n len_ofl_mdl = {len_ofl_mdl}")
print(f"\n---> Last offline train date = {last_tr_dt}\n")

training_data = train.filter(pl.col("date_id").le(last_tr_dt))
validation_data   = train.filter(pl.col("date_id").gt(last_tr_dt))

training_data = training_data.collect()
training_data = training_data.drop("partition_id")

# 创建存储训练集和验证集的文件夹
os.makedirs("train_folder", exist_ok=True)
os.makedirs("val_folder", exist_ok=True)

# 将训练数据进一步划分为 8 个部分
train_split_size = len(training_data) // 8
for i in range(8):
    start_idx = i * train_split_size
    # 最后一部分可能包含剩余数据
    end_idx = (i + 1) * train_split_size if i < 7 else len(training_data)
    split_data = training_data[start_idx:end_idx]
    folder_path = f"train_folder/partition_id={i+1}"
    os.makedirs(folder_path, exist_ok=True)
    
    # 使用 Polars 直接保存数据，而不是转换为 Pandas
    split_data.write_parquet(f"{folder_path}/train_data_0.parquet")

del training_data
gc.collect()

validation_data = validation_data.collect()
validation_data = validation_data.drop('partition_id')
# 将验证数据保存到两个文件夹
val_split_size = len(validation_data) // 2
for i in range(2):
    start_idx = i * val_split_size
    end_idx = (i + 1) * val_split_size if i < 1 else len(validation_data)
    
    split_data = validation_data[start_idx:end_idx]
    folder_path = f"val_folder/partition_id={i+1}"
    os.makedirs(folder_path, exist_ok=True)
    
    # 保存验证数据到对应文件夹
    split_data.write_parquet(f"{folder_path}/val_data_0.parquet")
del validation_data
gc.collect()
