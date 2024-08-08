# -*- coding: utf-8 -*-

import os

import time
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from generator import TimeSeriesDataGenerator



class LTSF_Linear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_Linear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for _ in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual


class LTSF_DLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, kernel_size, individual, feature_size):
        super(LTSF_DLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter(
                    (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter(
                    (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Trend.weight = torch.nn.Parameter(
                (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter(
                (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))

    def forward(self, x):
        trend_init, seasonal_init = self.decompsition(x)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.forcast_size],
                dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.forcast_size],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class LTSF_NLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x


class TrainSeq(object):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss,
        optimizer: torch.optim,
        epochs: int = 50,
        window: int = 60,
        forcast: int = 30,
        batch: int = 32,
        scaling: bool = False,
        device: str = 'cpu',
        save_path: str = "./",
        verbose: int = 0,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.window = window
        self.forcast = forcast
        self.batch = batch
        self.scaling = scaling
        self.device = device
        self.save_path = save_path
        self.verbose = verbose
        self.seed = seed
        self.log = pd.DataFrame(columns=["train_loss", "valid_loss", "test_loss"])

    def _train(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        self.model.train()
        loss_list = []
        for X, real, _ in dataloader:
            self.optimizer.zero_grad()
            pred = self.model(X.to(self.device))
            loss = self.criterion(pred, real.to(self.device))
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        return np.mean(loss_list)

    @torch.no_grad()
    def _test(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        self.model.eval()
        loss_list = []
        for X, real, _ in dataloader:
            pred = self.model(X.to(self.device))
            loss = self.criterion(pred, real.to(self.device))
            loss_list.append(loss.item())
        return np.mean(loss_list)

    def _get_dataloader(
        self,
        data: pd.DataFrame,
    ) -> tuple:
        # Data Scaling
        if self.scaling:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
            data = pd.DataFrame(
                data=self.scaler.transform(data), index=data.index, columns=data.columns)

        # Gen Dataset
        dataset = TimeSeriesDataset(
            data.values.astype(np.float32),
            data.index.values.astype(np.int64), self.window, self.forcast)

        _train_len = int(len(dataset) * 0.6)
        _valid_len = _train_len + int(len(dataset) * 0.2)
        _indices = list(range(len(dataset)))

        train_dataset = torch.utils.data.Subset(dataset, _indices[:_train_len])
        valid_dataset = torch.utils.data.Subset(dataset, _indices[_train_len:_valid_len])
        test_dataset = torch.utils.data.Subset(dataset, _indices[_valid_len:])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader

    def train(
        self,
        train_dataloader: torch.utils.data.dataloader.DataLoader,
        valid_dataloader: torch.utils.data.dataloader.DataLoader,
        test_dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        best_loss = np.inf
        for epoch in range(1, self.epochs + 1):
            required_time = time.time()
            train_loss = self._train(train_dataloader)
            valid_loss = self._test(valid_dataloader)
            test_loss = self._test(test_dataloader)

            required_time = time.time() - required_time
            self.log.loc[epoch] = {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "required_time": required_time,
            }

            if self.verbose:
                print(
                    f"epoch={epoch:>3},",
                    f"train_loss={train_loss:.5f},",
                    f"valid_loss={valid_loss:.5f},",
                    f"test_loss={test_loss:.5f},",
                    f"required_time={required_time:.2f}sec",
                    flush=True)

            if valid_loss < best_loss:
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(self.model, os.path.join(self.save_path, "model.pth"))

                if self.verbose:
                    print(
                        f"    Model Saved: best_loss={best_loss:.5f},",
                        f"valid_loss={valid_loss:.5f},",
                        f"test_loss={test_loss:.5f}",
                        flush=True)
                best_loss = valid_loss

        self.model = torch.load(os.path.join(self.save_path, "model.pth"))
        test_loss = self._test(test_dataloader)
        print(f"Best Model Result: test_loss={test_loss:.5f}", flush=True)

    def run(
        self,
        data: pd.DataFrame,
    ) -> None:
        total_required_time = time.time()

        # seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # preprocessing
        train_dataloader, valid_dataloader, test_dataloader = self._get_dataloader(data)

        # train
        if self.verbose:
            print("Model Train Start!", flush=True)
        self.train(train_dataloader, valid_dataloader, test_dataloader)
        self.log.to_csv(os.path.join(self.save_path, "log.csv"), encoding="euc-kr")    # Change encoding if you want.
        if self.verbose:
            print(f"Model Train End! total_required_time={time.time() - total_required_time:.2f}sec", flush=True)

        test_x, test_y, test_date = [], [], []
        for x, y, d in test_dataloader:
            test_x.append(x.detach().cpu().squeeze().numpy())
            test_y.append(y.detach().cpu().squeeze().numpy())
            test_date.append(np.vectorize(lambda x: np.datetime64(int(x), 'ns'))(d.numpy()))
        test_x = np.vstack(test_x)
        test_y = np.vstack(test_y)
        test_date = np.vstack(test_date)

        if self.scaling:
            test_x = self.scaler.inverse_transform(test_x)
            test_y = self.scaler.inverse_transform(test_y)

        return self.model, test_x, test_y, test_date

    @torch.no_grad()
    def infer(
        self,
        data: np.ndarray,
        real: np.ndarray | None = None,
    ):
        required_time = time.time()
        self.model.eval()
        res = []
        for i in range(data.shape[0] // self.batch + 1):    # First dim must be batch dim.
            X = torch.Tensor(data[i * self.batch: (i + 1) * self.batch]).unsqueeze(-1)
            res.append(self.model(X.to(self.device)).detach().cpu().squeeze().numpy())
        res = np.vstack(res)
        if self.scaling:
            res = self.scaler.inverse_transform(res)
        if self.verbose and real is not None:
            for i in range(self.forcast):
                print(f"Predict {i+1} min After:",
                      f"r2_score={r2_score(real[:,i], res[:,i]):.4f}",
                      f"RMSE={np.sqrt(mean_squared_error(real[:,i], res[:,i])):.4f}")
            print(f"required_time={time.time() - required_time:.2f}sec", flush=True)
        return res


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, date, window_size, forecast_size):
        self.data = data
        self.date = date
        self.window_size = window_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.forecast_size]
        d = self.date[idx + self.window_size:idx + self.window_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(d)


def get_result(
    pred: np.ndarray,
    real: np.ndarray,
    date: np.ndarray,
    model_name: str = "LinearRegression",
    minutes: list = [1, 5, 15, 30],
):
    fig, axes = plt.subplots(len(minutes), 2, figsize=(20, 5 * len(minutes)))
    fig.suptitle(f'{model_name} Model Time Series Prediction \n', fontsize=20)

    axes[0, 0].set_title('Entire Data', fontsize=15)
    axes[0, 1].set_title('Last 1 Day Data', fontsize=15)
    for i, minute in enumerate(minutes):
        res = pd.DataFrame({f"real_{minute}": real[:, i], f"pred_{minute}": pred[:, i]}, index=date[:, i])

        axes[i, 0].set_ylabel(f'Predict {minute} Min After', labelpad=15, fontsize=15)

        axes[i, 0].plot(res, label=["real", "pred"])
        axes[i, 0].legend(loc='upper left', fontsize=10)

        axes[i, 1].plot(res, label=["real", "pred"])
        axes[i, 1].legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()

    return None


if __name__ == "__main__":

    # Settings
    # # Model Setting
    lr = 0.001
    epochs = 50
    window = 60
    forcast = 30
    batch = 32
    scaling = False
    device = "cuda:0"
    save_path = "./model"
    verbose = 1
    seed = 42
    model_name = "DLinear"
    minutes = [1, 5, 15, 30]

    # # Data Setting
    date_range = {
        'start_time': '2024-01-01 00:00:00',
        'end_time': '2024-01-31 23:59:59',
        'freq': 'min',
    }

    clipping_range = {
        'min_bound': 0.,
        'max_bound': 500.,
    }

    pattern = {
        'Basic': 2,        # -1(None, Default), 1(Linear), 2(Sin), 3(Uniform), 4(Normal), 5(Server), 6(User)
        'Vibration': 1,    # -1(None, Default), 1(Noraml), 2(Uniform)
        'Abnoraml': 1,     # -1(None, Default), 1(1% Point Only), 2(2.5% Point, 2.5% Pattern),
                           # 3(5% Point, 5% Pattern), 4(12.5% Point, 12.5% Pattern), 5(25% Point, 25% Pattern)
    }


    # Execution Part
    # # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"사용 가능한 GPU의 개수: {num_gpus}")
    else:
        print("CUDA를 사용할 수 없습니다.")

    # # load model
    model = LTSF_DLinear(
        window_size=window,
        forcast_size=forcast,
        kernel_size=forcast + 1,
        individual=False,
        feature_size=1,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train = TrainSeq(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        window=window,
        forcast=forcast,
        batch=batch,
        scaling=scaling,
        device=device,
        save_path=save_path,
        verbose=verbose,
        seed=seed,
    )

    # # get data
    generator = TimeSeriesDataGenerator(date_range=date_range, clipping_range=clipping_range)
    data = generator.run(pattern=pattern, abnormal=False, plot=False, seed=seed)
    data = data.set_index(["datetime"])

    # # train
    model, test_x, test_y, test_date = train.run(data)

    # # infer
    pred = train.infer(test_x)

    # # plot
    get_result(pred, test_y, test_date, model_name, minutes)
