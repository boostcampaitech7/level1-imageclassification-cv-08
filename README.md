
# 🏆 Sketch Image Data Classification

## 🥇 팀 구성원
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/bogeoung"><img src="https://avatars.githubusercontent.com/u/50127209?v=4?s=100" width="100px;" alt=""/><br /><sub><b>김보경</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/SangphilPark"><img src="https://avatars.githubusercontent.com/u/81211140?v=4?s=100" width="100px;" alt=""/><br /><sub><b>박상필</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/LTSGOD"><img src="https://avatars.githubusercontent.com/u/78635028?v=4?s=100" width="100px;" alt=""/><br /><sub><b>이태순</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/d-a-d-a"><img src="https://avatars.githubusercontent.com/u/109848297?v=4?s=100" width="100px;" alt=""/><br /><sub><b>임현명</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/CheonJiEun"><img src="https://avatars.githubusercontent.com/u/53997172?v=4?s=100" width="100px;" alt=""/><br /><sub><b>천지은</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br />

## 프로젝트 소개
Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.

이번 프로젝트는 `부스트캠프 AI Tech 7기` CV 트랙 내에서 진행된 대회이며 Accuracy를 통해 최종평가를 진행하였습니다.

<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.09.10 ~ 2024.09.26

<br />

## 🥈 프로젝트 결과
- Private 리더보드에서 최종적으로 아래와 같은 결과를 얻었습니다.

<br />

## 🥉 데이터셋 구조
```
 data/
 ├── train.csv
 ├── test.csv
 ├── test
 │   └─ images
 └── train
     └── n01443537 ~ n13054560
        └─ images
 
```
이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
- 전체 이미지 수 : 25035
- 전체 클래스 수 : 500
- 한 클래스당 사진의 개수 : 30장 
- 이미지 크기 : (200~800, 200~800)
- 분류 클래스 : 개, 고양이, 거미, 뱀, 판다, 자동차, 기타, 건물 등
- 전체 데이터 중 학습데이터 60%(15000), 평가데이터 40%(10000)로 사용

<br />

## 🥉 프로젝트 구조
```
project/
├── .gitignore
├── Augmentation
│    ├─AutoAugGray.py
│    ├─AutoAugRGB.py
│    └─test
├── configs
│    └─config.py
├── data
│    └─data_loader.py
├── models
│    └─model_selector.py
├── optimizers
│    └─optimizer.py
├── schedulers
│    └─scheduler.py
├── trainers
│    ├─inference.py
│    ├─loss.py
│    ├─test_runner.py
│    └─trainer.py
├── utils
│    └─utils.py
└── main.py
```

#### 1) `Augmentation`
- 데이터셋을 읽고 RGB로 전환하거나 Grayscale로 전환하는 전처리를 진행하는 클래스를 구현
#### 2) `configs`
- 이미지 분류에 사용될 수 있는 다양한 hyperparameter들을 설정
- Model, Training, Augmentation, Optimizer, Scheduler 설정 구현
#### 3) `data`
- 데이터를 받아 초기 전처리를 처리한 후 다양한 증강 기법을 정의하는 파일 
- autoaugment, cutout, cutmix, mixup, translate, shear 구현
#### 4) `models`
- Timm Library로부터 데이터를 받아 연산을 처리한 후 결과 값을 내는 Model 클래스를 구현하는 파일 
#### 5) `optimizers`
- 학습에 사용할 다양한 Optimizer들을 정의한 파일
- adam, SGD, adadelta, adagrad, adamw, sparseadam, adamax, 등 구현
#### 6) `schedulers`
- Learning rate를 조절할 다양한 Scheduler들을 정의한 파일
- Step, Reduce, Cosine, Multistep, Lambda, Exponential, Cyclic, Onecycle 구현
#### 7) `inference.py`
- 학습 완료된 모델을 통해 test set 에 대한 예측 값을 구하고 이를 .csv 형식으로 저장하는 파일 
#### 8) `loss.py`
- 이미지 분류에 사용될 수 있는 다양한 Loss 들을 정의한 파일
- Cross Entropy, Focal Loss, Label Smoothing Loss, Asymmetric Loss 구현
#### 9) `trainer.py`
- 학습을 진행하고 검증을 거쳐서 학습 데이터 정확도를 내놓는 과정을 정의한 파일
#### 10) `utils.py`
- 학습을 진행하고 검증을 거쳐서 학습 데이터 정확도에 따라 early stopping을 정의하고 체크포인트를 생성해 loss 그래프를 출력하는 파일
#### 11) `main.py`
- 앞에 정의된 파일들을 Config를 기반으로 전체적으로 실행시켜서 학습을 진행하고 결과를 내놓는 파일

<br />

## ⚙️ 설치

#### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

pandas==2.1.4
matplotlib==3.8.4
seaborn==0.13.2
Pillow==10.3.0
numpy==1.26.3
timm==0.9.16
albumentations==1.4.4
tqdm==4.66.1
scikit-learn==1.4.2
opencv-python==4.9.0.80
python==3.10.13

Install dependencies: `pip install -r requirements.txt`

<br />

## 🚀 빠른 시작
#### Train
`python main.py`

이 외 다양한 학습 방법은 `🥉프로젝트 구조/4) train.py`를 참고해주세요!

<br />

## 🏅 Wrap-Up Report   
- [Wrap-Up Report👑]
