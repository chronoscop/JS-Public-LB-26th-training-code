import polars as pl
import pandas as pd
import numpy as np
import json
import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, callbacks
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold


dataaddr = "./dataset"


# 设置随机种子
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(seed=2025)


# 自定义加权 zero-mean R² 损失函数
def weighted_zero_mean_r2_loss(y_true, y_pred, weights):
    y_true_mean = torch.sum(weights * y_true) / torch.sum(weights)
    y_pred_mean = torch.sum(weights * y_pred) / torch.sum(weights)

    numerator = torch.sum(weights * (y_true - y_pred) ** 2)
    denominator = torch.sum(weights * (y_true - y_true_mean) ** 2)

    r2 = 1 - (numerator / denominator)
    return -r2  # 返回负的 R² 作为损失函数


# 自定义评估指标
def custom_metric(y_true, y_pred, weight):
    y_true_mean = np.sum(weight * y_true) / np.sum(weight)
    y_pred_mean = np.sum(weight * y_pred) / np.sum(weight)

    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    denominator = np.sum(weight * (y_true - y_true_mean) ** 2)

    weighted_r2 = 1 - (numerator / denominator)
    return weighted_r2


def r2_val(y_true, y_pred, sample_weight):
    # 确保输入为 NumPy 数组
    y_true = np.asarray(y_true).flatten()  # 转换为一维数组
    y_pred = np.asarray(y_pred).flatten()  # 转换为一维数组
    sample_weight = np.asarray(sample_weight).flatten()  # 转换为一维数组

    # 检查形状是否一致
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")

    if sample_weight.shape not in (y_pred.shape, (y_pred.shape[0],)):
        raise ValueError("sample_weight must have the same shape as y_pred or be 1D with length equal to y_pred.")

    # 计算 R²
    y_true_mean = np.average(y_true, weights=sample_weight)
    numerator = np.average((y_pred - y_true) ** 2, weights=sample_weight)
    denominator = np.average((y_true - y_true_mean) ** 2, weights=sample_weight)
    r2 = 1 - (numerator / denominator)

    return r2


print("< read parquet >")
weights = []
train = pd.DataFrame()  # 初始化

# 读取多个分区的训练数据
for i in range(0, 10):
    print(f"Reading partition {i}...")
    current_train = pl.read_parquet(
        f"{dataaddr}/train.parquet/partition_id={i}/part-0.parquet"
    ).to_pandas()

    # 随机抽样97%的数据
    current_train = current_train.sample(frac=0.97, random_state=2025)
    weights += list(current_train['weight'].values)
    current_train.drop(['weight'], axis=1, inplace=True)

    # 合并当前数据到主数据框
    train = pd.concat([train, current_train], ignore_index=True)

    del current_train
    gc.collect()
weights = np.array(weights)

print(f"train.shape: {train.shape}")

print("< get X,y >")
# 获取特征列
cols = ['symbol_id', 'time_id'] + [f'feature_0{i}' if i < 10 else f'feature_{i}' for i in range(79)]

X = train[cols].fillna(3).values
y = train['responder_6'].values
del train
gc.collect()

print("< train test split >")

# split = 300000  # 划分数据集的分割点（约2%）
# # 划分训练集和测试集
# train_X, train_y, val_X, val_y, train_weight, val_weight = (
#     X[:-split], y[:-split], X[-split:], y[-split:], weights[:-split], weights[-split:]
# )
# split_start = 1000000  # 验证集的起始点
# split_end = 600000    # 验证集的结束点
#
# # 划分训练集和测试集
# train_X, train_y, train_weight = (
#     X[:-split_start], y[:-split_start], weights[:-split_start]
# )
# val_X, val_y, val_weight = (
#     X[-split_start:-split_end], y[-split_start:-split_end], weights[-split_start:-split_end]
# )

split_ratio = 0.001  # 验证集占总数据的比例
k = int(1 / split_ratio)  # 每隔 k 个样本取一个作为验证集
# 创建索引数组
indices = np.arange(len(X))
# 创建训练集和验证集的索引
train_indices = []
val_indices = []
i = 0
while i < len(indices):
    if i % k == 0:
        # 连续取 100 个样本加入验证集
        val_indices.extend(indices[i:i+100])
        i += 100
    else:
        train_indices.append(indices[i])
        i += 1

# 使用索引划分数据集
train_X = X[train_indices]
train_y = y[train_indices]
train_weight = weights[train_indices]
val_X = X[val_indices]
val_y = y[val_indices]
val_weight = weights[val_indices]

print(f"train_X.shape:{train_X.shape},test_X.shape:{val_X.shape}")  # 打印训练和测试集的形状


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = X
        self.y = y
        self.weights = weights

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]]), torch.FloatTensor([self.weights[idx]])


# 定义 Gaussian Noise 层
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.035):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


# 定义神经网络模型
class SupervisedAutoencoder(LightningModule):
    def __init__(self, input_dim, encoder_dim, hidden_dims, dropouts, lr=0.001, weight_decay=0.0, ls=0.0):
        super().__init__()
        self.save_hyperparameters()
        # Encoder
        self.encoder = nn.Sequential(
            GaussianNoise(dropouts[0]),  # 添加高斯噪声层
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Dropout(dropouts[1]),
            nn.Linear(encoder_dim, input_dim)
        )
        # MLP (after concatenation)
        mlp_layers = []
        in_dim = input_dim + encoder_dim  # Concatenated input and encoder output
        mlp_layers.append(nn.BatchNorm1d(in_dim))
        mlp_layers.append(nn.Dropout(dropouts[2]))
        for i, hidden_dim in enumerate(hidden_dims):
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.SiLU())
            if i + 3 < len(dropouts):  # Adjust dropout indexing
                mlp_layers.append(nn.Dropout(dropouts[i + 3]))  # Add dropout layers accordingly
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        mlp_layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*mlp_layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.ls = ls  # Label smoothing parameter
        self.validation_step_outputs = []

    def forward(self, x):
        x0 = x  # Original input
        # Encoder
        z = self.encoder(x)
        # Decoder
        x_hat = self.decoder(z)
        # Concatenate x0 and z
        x_and_z = torch.cat([x0, z], dim=1)
        # MLP
        y_hat = 5 * self.mlp(x_and_z).squeeze(-1)
        return x_hat, y_hat

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        x_hat, y_hat = self(x)
        # Reconstruction loss
        rec_loss = F.mse_loss(x_hat, x)
        # Prediction loss
        y = y.view(-1)
        y_hat = y_hat.view(-1)
        pred_loss = F.mse_loss(y_hat, y, reduction='none') * w.view(-1)
        pred_loss = pred_loss.mean()
        # Total loss
        total_loss = rec_loss + pred_loss
        self.log('train_rec_loss', rec_loss, prog_bar=False)
        self.log('train_pred_loss', pred_loss, prog_bar=False)
        self.log('train_total_loss', total_loss, prog_bar=True)
        return pred_loss  # For early stopping monitor only pred_loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        x_hat, y_hat = self(x)
        # Reconstruction loss
        rec_loss = F.mse_loss(x_hat, x)
        # Prediction loss
        y = y.view(-1)
        y_hat = y_hat.view(-1)
        pred_loss = F.mse_loss(y_hat, y, reduction='none') * w.view(-1)
        pred_loss = pred_loss.mean()
        # Total loss
        total_loss = rec_loss + pred_loss
        self.log('val_rec_loss', rec_loss, prog_bar=False)
        self.log('val_pred_loss', pred_loss, prog_bar=False)
        self.log('val_total_loss', total_loss, prog_bar=True)
        self.validation_step_outputs.append((y_hat, y, w))
        return pred_loss  # For early stopping monitor only pred_loss

    def on_validation_epoch_end(self):
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        y_hat = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
        val_r_square = r2_val(y, y_hat, weights)
        self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_pred_loss',  # 监控训练损失
                'mode': 'min',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch  # 当前周期
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in
                   self.trainer.logged_metrics.items()}  # 记录的指标
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}  # 格式化指标
        print(f"Epoch {epoch}: {formatted_metrics}")  # 打印指标


# 创建数据集
train_dataset = CustomDataset(train_X, train_y, train_weight)
val_dataset = CustomDataset(val_X, val_y, val_weight)

# 创建早停回调
early_stop_callback = EarlyStopping(
    monitor='val_pred_loss',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='min'
)

# 定义ModelCheckpoint回调
checkpoint_callback = callbacks.ModelCheckpoint(
    monitor='val_pred_loss',
    dirpath='./5层/',
    filename='{epoch:02d}-{val_pred_loss:.5f}',
    save_top_k=10,
    mode='min'
)

# 模型超参数
input_dim = train_X.shape[1]
encoder_dim = 256
hidden_dims = [96, 896, 448, 448, 256,]
dropouts = [0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.271, 0.438]  # hidden_dims+3

# 训练模型
model = SupervisedAutoencoder(input_dim=input_dim, encoder_dim=encoder_dim, hidden_dims=hidden_dims, dropouts=dropouts, lr=1e-3)
trainer = Trainer(max_epochs=100, accelerator='gpu', devices=1,
                  callbacks=[checkpoint_callback, early_stop_callback])  # 使用 GPU 训练
trainer.fit(model, DataLoader(train_dataset, batch_size=64000, num_workers=8),
            DataLoader(val_dataset, batch_size=64000, num_workers=8))

