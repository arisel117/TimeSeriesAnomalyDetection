import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt





class TimeSeriesDataGenerator(object):
    def __init__(self, ) -> None:
        pass


    def run(
        self,
        trend: list,
        date_range: dict = {
            'start_time': '2024-01-01 00:00:00',
            'end_time': '2024-12-31 23:59:59',
            'freq': 'min', },
        clipping_range: dict = {
            'min_bound': 0.,
            'max_bound': 500., },
        plot: bool = False,
        seed: int = 42,
    ) -> pd.DataFrame:
        self.trend = trend
        self.date_range = date_range
        self.clipping_range = clipping_range

        np.random.seed(seed)
        data = self._get_baseline()
        data = self._get_trend(data)

        data['value'] = data['value'].where(
            data['value'] > self.clipping_range['min_bound'], self.clipping_range['min_bound'])
        data['value'] = data['value'].where(
            data['value'] < self.clipping_range['max_bound'], self.clipping_range['max_bound'])

        data = data.sort_values(['datetime']).reset_index(drop=True)

        if plot:
            self.plot_data(data)
        return data


    def _get_trend(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        for _trend in self.trend:
            method = _trend['method'].lower()
            start = _trend['start_value'] if 'start_value' in _trend else 0.
            end = _trend['end_value'] if 'end_value' in _trend else 0.
            freq = _trend['freq'].lower() if 'freq' in _trend else 'min'
            range_value = _trend['range_value'] if 'range_value' in _trend else 1
            cycle = _trend['cycle_value'] if 'cycle_value' in _trend else 1
            custom_value = _trend['custom_value'] if 'custom_value' in _trend else []


            if method == 'abnormal_point':
                data = data.sort_values(['datetime']).reset_index(drop=True)
                data = self._abnormal_point(
                    data=data,
                    abnormal_ratio=_trend['abnormal_ratio'],
                    abnormal_range=_trend['abnormal_range'])
                continue
            elif method == 'abnormal_pattern':
                data = data.sort_values(['datetime']).reset_index(drop=True)
                data = self._abnormal_pattern(
                    data=data,
                    abnormal_ratio=_trend['abnormal_ratio'],
                    abnormal_range=_trend['abnormal_range'],
                    abnormal_dist=_trend['abnormal_dist'])
                continue

            if freq in ['year', 'y', 'a']:
                data['freq_time'] = data['datetime'].dt.year
            elif freq in ['month', 'm']:
                data['freq_time'] = data['datetime'].dt.month
            elif freq in ['weak', 'w', 'day', 'd']:    # Day is also treated weakly.
                data['freq_time'] = data['datetime'].dt.weekday
            elif freq in ['hour', 'h']:
                data['freq_time'] = data['datetime'].dt.hour
            elif freq in ['min']:
                data['freq_time'] = data['datetime'].dt.minute
            else:
                raise ValueError("freq must one of ['year', 'month', 'weak', 'hour', 'min'].")
            extd = pd.DataFrame(sorted(data['freq_time'].unique()), columns=['freq_time'])
            _len = len(extd)

            if method == 'linear':
                extd['cache'] = self._linear(data_len=_len, start=start, end=end)
            elif method == 'sin':
                extd['cache'] = self._linear(data_len=_len, start=start, end=end)
                extd['cache'] += self._sin(data_len=_len, data_range=range_value, cycle=cycle)
            elif method == 'uniform':
                extd['cache'] = self._linear(data_len=_len, start=start, end=end)
                extd['cache'] += self._uniform(data_len=_len, data_range=range_value)
            elif method == 'normal':
                extd['cache'] = self._linear(data_len=_len, start=start, end=end)
                extd['cache'] += self._normal(data_len=_len, data_range=range_value)
            elif method == 'custom':
                extd['cache'] = self._linear(data_len=_len, start=start, end=end)
                extd['cache'] += self._custom(data_len=_len, custom_value=custom_value)
            else:
                raise ValueError("method must be one of \
                    ('linear', 'sin', 'uniform', 'normal', 'custom', \
                    'abnormal_point', 'abnormal_pattern').")

            data = pd.merge(data, extd, on='freq_time')
            data['value'] += data['cache']
            data.drop(columns=['freq_time', 'cache'], inplace=True)

        return data


    def plot_data(
        self,
        data: pd.DataFrame,
    ) -> None:

        _, axes = plt.subplots(2, 2, figsize=(20, 10))

        sub_data = data.copy()
        axes[0, 0].plot(sub_data.set_index(['datetime']))
        axes[0, 0].set_title('Entire Data')

        sub_data = data[data['datetime'].dt.month.isin([1])]
        axes[0, 1].plot(sub_data.set_index(['datetime']))
        axes[0, 1].set_title('1 Month Data')

        sub_data = data[data['datetime'].dt.month.isin([1])].reset_index(drop=True)
        sub_data = sub_data[sub_data['datetime'].dt.day.isin([1, 2, 3, 4, 5, 6, 7])]
        axes[1, 0].plot(sub_data.set_index(['datetime']))
        axes[1, 0].set_title('1 Week Data')

        sub_data = data[data['datetime'].dt.month.isin([1])].reset_index(drop=True)
        sub_data = sub_data[sub_data['datetime'].dt.day.isin([1])]
        axes[1, 1].plot(sub_data.set_index(['datetime']))
        axes[1, 1].set_title('1 Day Data')

        plt.tight_layout()
        plt.show()

        return None


    def _get_baseline(
        self,
    ) -> pd.DataFrame:
        data = pd.DataFrame(
            pd.date_range(
                start=self.date_range['start_time'],
                end=self.date_range['end_time'],
                freq=self.date_range['freq'],),
            columns=['datetime'],
        )
        data['value'] = 0

        return data


    def _linear(
        self,
        data_len: int = 100,
        start: int = 1,
        end: int = 100,
    ) -> np.array:
        return np.linspace(start=start, stop=end, num=data_len)


    def _sin(
        self,
        data_len: int = 100,
        data_range: int = 10,
        cycle: int = 1,
    ) -> np.array:
        return np.sin(np.linspace(start=0, stop=cycle * 2 * np.pi, num=data_len)) * data_range
        # return np.sin(np.linspace(start=0, stop=(data_len // cycle) * 2 * np.pi, num=data_len)) * data_range


    def _uniform(
        self,
        data_len: int = 100,
        data_range: int = 10,
    ) -> np.array:
        return np.random.uniform(low=-abs(data_range // 2), high=abs(data_range // 2), size=data_len)


    def _normal(
        self,
        data_len: int = 100,
        data_range: int = 10,
    ) -> np.array:
        return np.random.normal(loc=0, scale=data_range / 2, size=data_len)


    def _custom(
        self,
        data_len: int = 100,
        custom_value: list = [],
    ) -> np.array:
        custom_value = custom_value.copy()

        if len(custom_value) > data_len:
            custom_value = custom_value[:data_len]
        elif len(custom_value) == data_len:
            custom_value = custom_value
        else:
            custom_value = custom_value + [0] * (data_len - len(custom_value))

        return np.array(custom_value)


    def _abnormal_point(
        self,
        data: pd.DataFrame,
        abnormal_ratio: float,
        abnormal_range: tuple,
    ) -> pd.DataFrame:
        data = data.copy()

        abnormal_cnt = int(len(data) * abnormal_ratio)
        extd = pd.DataFrame(
            data=np.random.choice(
                [abs(abnormal_range[0]), -abs(abnormal_range[0])],
                size=abnormal_cnt) + self._normal(
                    data_len=abnormal_cnt, data_range=abnormal_range[1]),
            index=np.random.choice(
                [i for i in range(len(data))],
                size=abnormal_cnt,
                replace=False),
            columns=['cache'],
        )

        data = data.join(extd, how='left')
        data['value'] += data['cache'].fillna(0)
        data.drop(columns=['cache'], inplace=True)

        return data


    def _get_abnormal(
        self,
        target_sum: int,
        min_value: int,
        max_value: int,
    ) -> list:
        if (target_sum < min_value) or (target_sum > max_value * (target_sum // min_value)):
            raise ValueError("'target_sum' must be greater than 'min_value'.")

        abnormals = deque()
        current_sum = 0

        while current_sum < target_sum:
            remaining_sum = target_sum - current_sum
            if remaining_sum < min_value:
                current_sum -= abnormals.popleft()
                continue
            elif remaining_sum > max_value:
                abnormals.append(np.random.randint(min_value, max_value))
                current_sum += abnormals[-1]
            else:
                abnormals.append(remaining_sum)
                break

        return list(abnormals)


    def _get_normal(
        self,
        target_sum: int,
        normals_len: int,
    ) -> list:

        normals = [0]
        if target_sum < normals_len:
            normals.extend([0] * (normals_len - target_sum))
            normals_len += 1 - len(normals)

        normals.extend(np.random.choice(
            [i for i in range(target_sum)],
            size=normals_len,
            replace=False).tolist())
        normals = sorted(normals + [target_sum])

        return [normals[i + 1] - normals[i] for i in range(len(normals) - 1)]


    def _abnormal_pattern(
        self,
        data: pd.DataFrame,
        abnormal_ratio: float,
        abnormal_range: tuple,
        abnormal_dist: tuple,
    ) -> pd.DataFrame:
        data = data.copy()

        min_value, max_value = sorted(abnormal_dist)
        abnormals = self._get_abnormal(
            target_sum=int(len(data) * abnormal_ratio),
            min_value=min_value,
            max_value=max_value)
        normals = self._get_normal(
            target_sum=len(data) - sum(abnormals),
            normals_len=len(abnormals))

        abnormals_center = np.random.choice(
            [abs(abnormal_range[0]), -abs(abnormal_range[0])],
            size=len(abnormals)) + self._normal(
            data_len=len(abnormals),
            data_range=abnormal_range[1])

        current_index = 0
        for a, b, _cen in zip(normals, abnormals, abnormals_center):
            current_index += a
            data.loc[current_index: current_index + b - 1, "value"] = _cen
            + self._normal(data_len=b, data_range=abnormal_range[1])
            current_index += b

        return data


    def get_sample_pattern(
        self,
        pattern: dict = {'Basic': -1, 'Vibration': -1, 'Abnoraml': -1, },
    ) -> list:
        trend = []
        trend += self._get_basic_pattern(pattern_idx=pattern['Basic'])
        trend += self._get_vibration_pattern(pattern_idx=pattern['Vibration'])
        trend += self._get_abnoraml_pattern(pattern_idx=pattern['Abnoraml'])

        return trend


    def _get_basic_pattern(
        self,
        pattern_idx: int
    ) -> list:
        if pattern_idx == -1:    # No Pattern (None, Default)
            trend = [
                {
                    'freq': 'year',
                    'method': 'linear',
                    'start_value': 0.,
                    'end_value': 0.,
                },
            ]
        elif pattern_idx == 1:    # Linear Pattern
            trend = [
                {
                    'freq': 'month',
                    'method': 'linear',
                    'start_value': 100.,
                    'end_value': 150.,
                },
            ]
        elif pattern_idx == 2:     # Sin Pattern
            trend = [
                {
                    'freq': 'month',
                    'method': 'linear',
                    'start_value': 100.,
                    'end_value': 150.,
                },
                {
                    'freq': 'month',
                    'method': 'sin',
                    'start_value': 0.,
                    'end_value': 0.,
                    'cycle_value': 2,
                    'range_value': 20.,
                },
                {
                    'freq': 'weak',
                    'method': 'sin',
                    'start_value': 0.,
                    'end_value': 0.,
                    'cycle_value': 1,
                    'range_value': 10.,
                },
                {
                    'freq': 'hour',
                    'method': 'sin',
                    'start_value': 0.,
                    'end_value': 0.,
                    'cycle_value': 3,
                    'range_value': 5.,
                },
                {
                    'freq': 'min',
                    'method': 'sin',
                    'start_value': 0.,
                    'end_value': 0.,
                    'cycle_value': 1,
                    'range_value': 3.,
                },
            ]
        elif pattern_idx == 3:     # Uniform Pattern
            trend = [
                {
                    'freq': 'month',
                    'method': 'linear',
                    'start_value': 100.,
                    'end_value': 150.,
                },
                {
                    'freq': 'month',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 20.,
                },
                {
                    'freq': 'weak',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 10.,
                },
                {
                    'freq': 'hour',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 5.,
                },
                {
                    'freq': 'min',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
            ]
        elif pattern_idx == 4:     # Normal Pattern
            trend = [
                {
                    'freq': 'month',
                    'method': 'linear',
                    'start_value': 100.,
                    'end_value': 150.,
                },
                {
                    'freq': 'month',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 20.,
                },
                {
                    'freq': 'weak',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 10.,
                },
                {
                    'freq': 'hour',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 5.,
                },
                {
                    'freq': 'min',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
            ]
        elif pattern_idx == 5:     # User Pattern
            trend = [
                {
                    'freq': 'month',
                    'method': 'linear',
                    'start_value': 100.,
                    'end_value': 150.,
                },
                {
                    'freq': 'month',
                    'method': 'sin',
                    'start_value': -5.,
                    'end_value': 5.,
                    'cycle_value': 1,
                    'range_value': 20.,
                },
                {
                    'freq': 'weak',
                    'method': 'custom',
                    'start_value': 0.,
                    'end_value': 0.,
                    'custom_value': [-6, -3, 2, 3, 4, 10, -10],
                },
                {
                    'freq': 'hour',
                    'method': 'custom',
                    'start_value': 0.,
                    'end_value': 0.,
                    'custom_value': [
                        -10, -10, -10, -10, -10, -10, -10, -8, -6, 0, 6, 10,
                        6, 4, 6, 5, 4, 0, -6, -8, -8, -10, -10, -10],
                },
            ]
        else:
            raise Exception("DataRangeError: Basic Pattern must be one of (-1, 1, 2, 3, 4, 5).")

        return trend


    def _get_vibration_pattern(
        self,
        pattern_idx: int
    ) -> list:
        if pattern_idx == -1:    # No Vibration (None, Default)
            trend = []
        elif pattern_idx == 1:    # 1% Abnormal (1% Point Only)
            trend = [
                {
                    'freq': 'year',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'month',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'weak',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'hour',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'min',
                    'method': 'uniform',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
            ]
        elif pattern_idx == 2:     # 5% Abnormal (2.5% Point, 2.5% Pattern)
            trend = [
                {
                    'freq': 'year',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'month',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'weak',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'hour',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
                {
                    'freq': 'min',
                    'method': 'normal',
                    'start_value': 0.,
                    'end_value': 0.,
                    'range_value': 3.,
                },
            ]
        else:
            raise Exception("DataRangeError: Vibration Pattern must be one of (-1, 1, 2).")

        return trend


    def _get_abnoraml_pattern(
        self,
        pattern_idx: int
    ) -> list:
        if pattern_idx == -1:    # No Abnormal (None, Default)
            trend = []
        elif pattern_idx == 1:    # 1% Abnormal (1% Point Only)
            trend = [
                {
                    'method': 'abnormal_point',
                    'abnormal_ratio': 0.01,
                    'abnormal_range': (400., 50.),
                },
            ]
        elif pattern_idx == 2:     # 5% Abnormal (2.5% Point, 2.5% Pattern)
            trend = [
                {
                    'method': 'abnormal_point',
                    'abnormal_ratio': 0.025,
                    'abnormal_range': (400., 50.),
                },
                {
                    'method': 'abnormal_pattern',
                    'abnormal_ratio': 0.025,
                    'abnormal_range': (400., 50.),
                    'abnormal_dist': (30, 2 * 24 * 60),
                },
            ]
        elif pattern_idx == 3:     # 10% Abnormal (5% Point, 5% Pattern)
            trend = [
                {
                    'method': 'abnormal_point',
                    'abnormal_ratio': 0.05,
                    'abnormal_range': (400., 50.),
                },
                {
                    'method': 'abnormal_pattern',
                    'abnormal_ratio': 0.05,
                    'abnormal_range': (400., 50.),
                    'abnormal_dist': (30, 2 * 24 * 60),
                },
            ]
        elif pattern_idx == 4:     # 25% Abnormal (12.5% Point, 12.5% Pattern)
            trend = [
                {
                    'method': 'abnormal_point',
                    'abnormal_ratio': 0.125,
                    'abnormal_range': (400., 50.),
                },
                {
                    'method': 'abnormal_pattern',
                    'abnormal_ratio': 0.125,
                    'abnormal_range': (400., 50.),
                    'abnormal_dist': (12 * 60, 2 * 24 * 60),
                },
            ]
        elif pattern_idx == 5:     # 50% Abnormal (25% Point, 25% Pattern)
            trend = [
                {
                    'method': 'abnormal_point',
                    'abnormal_ratio': 0.25,
                    'abnormal_range': (400., 50.),
                },
                {
                    'method': 'abnormal_pattern',
                    'abnormal_ratio': 0.25,
                    'abnormal_range': (400., 50.),
                    'abnormal_dist': (12 * 60, 2 * 24 * 60),
                },
            ]
        else:
            raise Exception("DataRangeError: Abnoraml Pattern must be one of (-1, 1, 2, 3, 4, 5).")

        return trend
