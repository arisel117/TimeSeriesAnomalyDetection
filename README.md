# 시계열 데이터의 이상 분석


## 참조 사이트
```
https://paperswithcode.com/task/time-series-anomaly-detection
```


## 선행 연구
- [TimeSeriesBench: An Industrial-Grade Benchmark for Time Series Anomaly Detection Models](https://arxiv.org/abs/2402.10802)
  - TimeSeriesBench라는 리더보드를 도입하여 시계열 이상 탐지 분야의 벤치마킹을 할 수 있는 시스템을 제안함
  - 여러 모델에 대한 벤치마크 결과를 표현함
    - [AR](https://books.google.co.kr/books?hl=ko&lr=&id=woaH_73s-MwC&oi=fnd&pg=PR13&dq=Peter+J+Rousseeuw+and+Annick+M+Leroy.+2005.+Robust+regression+and+outlier+detection&ots=TDnGKRzikV&sig=-NYc4UeEdey1IXnAgo8hHtRptas#v=onepage&q=Peter%20J%20Rousseeuw%20and%20Annick%20M%20Leroy.%202005.%20Robust%20regression%20and%20outlier%20detection&f=false) (2005): 파라미터 수가 작고, 성능이 우수함
    - [FCVAE](https://dl.acm.org/doi/abs/10.1145/3589334.3645710) (2024): 추론 속도가 빠르고, 성능이 우수함
    - [OFA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html) (2023): 사전 학습된 GPT를 백본으로 사용하여 파라미터 수가 많고, 성능이 우수함
    - [Anomaly Transformer](https://arxiv.org/abs/2110.02642) (2022): 파라미터의 수가 많고, 기대보다 성능이 낮게 나타남
    - 이 밖에도, LSTMAD, AE, FITS, Donut, TranAD, EncDecAD, TimesNet, SRCNN 등의 모델도 벤치마크됨

- MTAD: Tools and Benchmarks for Multivariate Time Series Anomaly Detection
- Making the End-User a Priority in Benchmarking: OrionBench for Unsupervised Time Series Anomaly Detection
- MOSPAT: AutoML based Model Selection and Parameter Tuning for Time Series Anomaly Detection
- Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

- LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
- XGBoost: A Scalable Tree Boosting System
- DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series
- Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
- Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model
- Time Series Anomaly Detection for Cyber-Physical Systems via Neural System Identification and Bayesian Filtering

- Glow: Generative Flow with Invertible 1x1 Convolutions
- Deep and Confident Prediction for Time Series at Uber
- A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data

