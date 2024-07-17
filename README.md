# 시계열 데이터의 이상 분석

<br/>  <br/>

* * *
## 관련분야 선행 연구 정리
  - 도메인(데이터셋)의 특성에 따라 적합한 모델은 다를 수 있으며, 신경망의 깊이가 깊다고 반드시 좋은 결과를 도출하진 않음
    - EDA와 같은 **데이터셋 자체에 대한 이해**가 선행적으로 이루어져야함
    - 단변량 vs 다변량, (학습/추론에 따른) 제한된 작업 환경 요건 등도 고려해야함
  - 적합한 모델이라 하더라도 **하이퍼 파라미터**에 따라 기울기 소실 혹은 폭주가 일어날 수 있으므로 적절한 해결 방법론들이 필요함
    - **활성 함수의 조정, 기울기 클리핑, 적절한 가중치 초기화, 배치 정규화** 등을 통해 다소 해결이 가능함
    - **AutoML**의 적용 (**[RayTune](https://docs.ray.io/en/latest/tune/index.html), [Auto-PyTorch](https://github.com/automl/Auto-PyTorch), [Optuna](https://optuna.org/), [AutoKeras](https://autokeras.com/), [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [AutoML for XGBoost](https://microsoft.github.io/FLAML/docs/Examples/AutoML-for-XGBoost/), [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html)** 등)
* * *

<br/>  <br/>


## 데이터 생성기 (v0.1)
  - [코드 링크](https://github.com/arisel117/TimeSeriesAnomalyDetection/blob/main/generator.py)
  - 샘플 데이터 추가 필요



* * *
### 참조 사이트
```
https://paperswithcode.com/task/time-series-anomaly-detection
```
* * *

<br/>  <br/>

* * *
### 참조 선행 연구
- [TimeSeriesBench: An Industrial-Grade Benchmark for Time Series Anomaly Detection Models](https://arxiv.org/abs/2402.10802)
  - TimeSeriesBench라는 리더보드를 도입하여 시계열 이상 탐지 분야의 **벤치마킹**을 할 수 있는 시스템을 제안함
  - 여러 모델에 대한 벤치마크 결과를 표현함
    - [AR](https://books.google.co.kr/books?hl=ko&lr=&id=woaH_73s-MwC&oi=fnd&pg=PR13&dq=Peter+J+Rousseeuw+and+Annick+M+Leroy.+2005.+Robust+regression+and+outlier+detection&ots=TDnGKRzikV&sig=-NYc4UeEdey1IXnAgo8hHtRptas#v=onepage&q=Peter%20J%20Rousseeuw%20and%20Annick%20M%20Leroy.%202005.%20Robust%20regression%20and%20outlier%20detection&f=false) (2005): 파라미터 수가 작고, 성능이 우수함
    - **[FCVAE](https://dl.acm.org/doi/abs/10.1145/3589334.3645710) (2024)**: 추론 속도가 빠르고, 성능이 우수함
    - **[OFA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html) (2023)**: 사전 학습된 GPT를 백본으로 사용하여 파라미터 수가 많고, 성능이 우수함
    - **[Anomaly Transformer](https://arxiv.org/abs/2110.02642) (2022)**: 파라미터의 수가 많고, 기대보다 성능이 낮게 나타남
    - 이 밖에도, LSTMAD, AE, FITS, Donut, TranAD, [EncDecAD](https://arxiv.org/abs/1607.00148), TimesNet, SRCNN 등의 모델도 벤치마크됨

- [MTAD: Tools and Benchmarks for Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2401.06175)
  - 5가지 데이터셋으로 12가지 모델을 벤치마크함
    - **KNN, AutoEncoder, LSTM, LSTM_VAE 모델이 우수한 결과**를 보임
    - **KNN은 대부분의 데이터셋에서 우수**했고, 최고 성능은 DL 모델에서 나타남
    - 이 밖에도, iForest, LODA, LOF, PCA, DAGMM, MAD_GAN, MSCRED, OmniAnomaly 등의 모델도 벤치마크됨
    - 모든 데이터셋에 **가장 적합한 모델은 없으며**, DL 모델의 경우 튜닝이 중요함

- [Making the End-User a Priority in Benchmarking: OrionBench for Unsupervised Time Series Anomaly Detection](https://arxiv.org/abs/2310.17748)
  - OrionBench라는 리더보드를 도입하여 시계열 이상 탐지 분야의 벤치마킹을 할 수 있는 시스템을 제안함
  - 14가지 데이터셋으로 12가지 모델을 벤치마크함
    - [AER](https://ieeexplore.ieee.org/abstract/document/10020857), [LSTM DT](https://dl.acm.org/doi/abs/10.1145/3219819.3219845), 모델이 비교적 우수한 결과를 보임
    - ARIMA, MP, LSTM AE, TadGAN, VAE, Dense AE, GANF, LNN, Azure AD, AT
  - 모든 데이터셋에 **가장 적합한 모델은 없음**

- **[MOSPAT: AutoML based Model Selection and Parameter Tuning for Time Series Anomaly Detection](https://arxiv.org/abs/2205.11755)**
  - 시계열 이상 탐지 모델의 매개변수 선택을 위한 End-to-End 시스템을 제안함
  - AutoML, Outlier, CUSUM, BOCPD, Statsig, MKDetector, Prophet 방법론을 비교함
    - **AutoML이 일관되게 우수한 성능**을 나타냄
    - Bayesian Optimization 방법을 통해 학습 시간을 단축

- [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642)
  - 일반적인 Transformer의 Attention Block 대신 Anomaly-Attention Block을 사용하여 보다 우수한 결과를 보임
  - 5개의 데이터셋에서 **LSTM, LSTM-VAE, BeatGAN**, Deep-SVDD, DAGMM 및 다양한 선행 연구들과 비교하여 일관되게 우수한 결과를 보임
    - 비교를 위한 다른 선행 연구들의 경우 데이터셋에 따라 좋은 결과를 보이는 모델이 다르고, 제안된 방법론만 모든 데이터셋에 대해 **일관되게 우수한 결과**를 보이고 있어 인상적임
    - 이 제안 방법론을 참조하는 후행 연구들에서는 보다 여러 데이터셋에서 모델의 종합적인 정확도를 벤치마크 하였는데, 정확도가 상대적으로 낮게 나타나는 것을 고려하면 도메인(데이터셋)의 특성에 따라 적절한 모델의 선정 및 튜닝이 무엇보다 중요 할 것으로 생각됨

- [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)
  - LSTM 네트워크 기반 Encoder-Decoder 구조(EncDec-AD)를 제안함
  - 모델의 구조적 특징을 생각해보면, 단변량 데이터보다 **다변량 데이터에 유용한 모델**로 생각됨
  - 다른 모델들은 이상 탐지를 하려면 시계열 예측이 가능해야 했지만, 제안 방법론은 **예측할 수 없는 시계열에서도 이상 징후를 탐지**한 부분이 인상적임
  - [다른 블로그](https://joungheekim.github.io/2020/11/14/code-review/)들에서 코드 리뷰를 한 기록이 많음
 
- **[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)**
  - Sparsity-aware Split Finding에 의해 **결측치가 있어도 학습이 가능**함
  - 분산 환경에서도 학습이 가능하고, CPU 캐시를 고려한 알고리즘이라 **데이터가 많아도 빠른 속도로 학습이 가능**함
  - 명확하게 모델의 **학습 결과를 설명 가능하다는 특징**이 있음
  - 논문의 인용수(4만↑)만 봐도 알 수 있듯이 **다양한 분야에서 사용되고, 정확도가 높은 것이 검증**된 모델이며, 패키지화 되어 있어 **사용성도 매우 편리**하여 baseline model로 사용이 가능 할 것으로 생각됨
  - 도메인 특성에 따라 튜닝이 필요한 만큼 [GridSearchCV](https://www.kaggle.com/code/phunter/xgboost-with-gridsearchcv)나 **[AutoML](https://microsoft.github.io/FLAML/docs/Examples/AutoML-for-XGBoost/)을 사용해서 튜닝**하는 방법이 매우 유용 할 것으로 생각됨

- [DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series](https://ieeexplore.ieee.org/document/8581424)
  - **결측치가 있어도 학습이 가능**하고, **비교적 작은 데이터셋에서도 모델을 학습**할 수 있음
  - 다양한 모델, 다양한 데이터셋에서 비교한 결과를 제공하고 있지는 않아서 다소 아쉬움
  - **구조가 정말 간단**하기 때문에 구현이 쉽고, 모델 파라미터 수가 작은 CNN의 특징을 고려하면 잘 튜닝되면 End(Edge)단으로 Porting해서 사용하기에도 좋을 것으로 생각됨

- [Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model](https://ieeexplore.ieee.org/document/9053558)
  - 5가지 데이터셋에서 VAE, LSTM-AD, ARMA 모델과 비교하여 우수한 결과를 보임
  - 제안된 VAE를 접목한 LSTM은 다른 방법론과 다르게 **저차원의 Embedding Layer를 사용**하는 것을 고려하면 다변량 데이터보다 **단변량 데이터에서 유용한 모델**로 생각됨

- [Time Series Anomaly Detection for Cyber-Physical Systems via Neural System Identification and Bayesian Filtering]()
  - 3가지 데이터셋에 Isolation Forest, Sparse-AE, EncDec-AD, LSTM-Pred, DAGMM, OmniAnomaly, USAD 라는 다양한 선행 연구들과 비교함
FFN, LSTM, 베이지안 필터링으로 구성된 NSIBF 모델이 우수한 결과를 보였으나, 보다 구체적인 모델 구조나 하이퍼파라미터들이 있으면 좋겠음

- **[Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)**
  - 보다 자세한 내용을 알기 위해서는 선행 연구를 살펴 볼 필요가 있음
    - **[Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770)** 논문의 Normalizing Flow 개념의 이해가 필요
    - **[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)** 논문의 GAN 모델의 이해가 필요
    - **[Density estimation using real nvp](https://arxiv.org/abs/1605.08803)** 논문의 real nvp 모델의 이해가 필요
    - NN의 마지막 레이어에 **zero initialization**을 해주면, 처음에는 identity function처럼 작동하고, DL을 학습하기에 좋다고 함! 새로운 지식!
    - Normalizing Flows에서 Temperature가 높으면 확률 분포가 균등하게 분포하게되어 출력의 무작위성으로 이어지고, **낮으면 출력이 조금 더 결정적(Deterministic)**으로 나타나게 되는 효과가 있음을 알 수 있음

- [Deep and Confident Prediction for Time Series at Uber](https://arxiv.org/abs/1709.01907)
  - Last-Day, QRF, LSTM 모델과 비교하여, AE-LSTM 구조에 Externel Features를 사용한 모델을 제안함
  - 제한된 Uber 데이터셋에서만 비교 분석된 부분이 다소 아쉬움

- [A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data](https://arxiv.org/abs/1811.08055)
  - 다층 구조의 Conv LSTM 모델을 AutoEncoder 형태로 쌓은 모델인 MSCRED(Multi-Scale Convolutional Recurrent Encoder-Decoder)을 제안함
  - OC-SVM, DAGMM, HA, ARMA, LSTM-ED, ConvLSTM 모델들과 비교하여 우수한 결과를 보임
  - 다변량 데이터에 대한 학습이 가능하고, 디테일한 모델 구조에 대한 설명이 있어 재구현은 충분히 가능 할 것으로 생각됨

* * *
