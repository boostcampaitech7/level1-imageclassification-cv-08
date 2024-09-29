
# 🏆 Sketch Image Data Classification

## 🥇 팀 구성원

#### 박재우, 이상진, 유희석, 정지훈, 천유동, 임용섭


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
- 이미지 크기 : (200-800, 200-800)
- 분류 클래스 : 개, 고양이, 거미, 뱀, 판다, 자동차, 기타, 건물 등
- 전체 데이터 중 학습데이터 60%(15000), 평가데이터 40%(10000)로 사용

<br />

## 🥉 프로젝트 구조
```
project/
|   config.yaml
|   main.py
|   README.md
|   requirements.txt
|   test.py
|   train.py
|
+---configs
|       config_manager.py
|
+---data
|       custom_dataset.py
|       cutmix.py
|       cutmix_loader.py
|       data_loader.py
|       transform_selector.py
|
+---EDA
|       Level_1_CV_08_EDA.pptx
|
+---models
|       model_selector.py
|
+---optimizers
|       optimizer.py
|
+---schedulers
|       customCosineWR.py
|       scheduler.py
|
+---trainers
|       inference_runner.py
|       loss.py
|       metric.py
|       test_runner.py
|       train_runner.py
|
\---utils
        utils.py
```

### 1) `configs`
- 설정 파일을 관리하는 폴더
- `config_manager.py`는 `config.yaml`을 불러와 학습 및 추론에 필요한 설정을 관리합니다.
### 2) `datas`
- 데이터 로딩 및 전처리를 담당하는 폴더
- `custom_dataset.py`에서는 커스텀 데이터셋을 정의하며, `cutmix.py`와 같은 데이터 증강 기법도 포함되어 있습니다.
### 3) `models`
- 모델 선택 및 초기화 로직을 정의하는 폴더
- `model_selector.py`에서 `timm 라이브러리`를 통해 다양한 사전 학습된 모델을 선택하고 사용할 수 있습니다.
### 4) `optimizers`
- 학습 중 Learning Rate 조절을 위한 스케줄러를 정의하는 폴더
- `optimizer.py`에서 `Adam`, `SGD`, `AdamW` 옵티마이저를 설정할 수 있습니다.
### 5) `schedulers`
- 학습 중 Learning Rate 조절을 위한 스케줄러를 정의하는 폴더
- `scheduler.py`는 다양한 학습률 조절 방법을 제공합니다.
### 6) `trainers`
- 학습과 추론에 필요한 주요 로직을 포함하는 폴더
- `train_runner.py`는 학습을 진행하는 클래스이며, `test_runner.py`는 모델 평가를 수행합니다.
### 7) `utils`
- 학습과 테스트 과정에서 사용되는 유틸리티 함수들을 정의한 폴더
- `utils.py`는 로깅, 체크포인트 저장 등 다양한 기능을 제공합니다.

<br />

## ⚙️ 설치

### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

```bash
pip install -r requirements.txt
```

- pandas==2.1.4
- matplotlib==3.8.4
- seaborn==0.13.2
- Pillow==10.3.0
- numpy==1.26.3
- timm==0.9.16
- albumentations==1.4.4
- tqdm==4.66.1
- scikit-learn==1.4.2
- opencv-python==4.9.0.80
- python==3.10.13

<br />

## 🚀 빠른 시작
### Main
```bash
python3 main.py
```
- `config.yaml` 파일을 수정한 후, 해당 명령어를 사용하여 학습과 추론을 모두 진행할 수 있습니다. 
### Train
```bash
python3 train.py --config_path ./config.yaml --epochs 10 --batch_size 32 --lr 0.0005 --use_cutmix
```
- `--config_path` : 설정 파일 경로 (기본값 : config.yaml)
- `--split_ratio` : 학습/검증 데이터 분할 비율 (기본값 : 0.2)
- `--use_cutmix` : CutMix 사용시 플래그 추가
- `--epochs` : 학습할 에폭 수 (기본값 : 5)
- `--lr` : 학습률 설정
- `--batch_size` : 배치 크기 설정
- `--img_size` : Resize 이미지 크기
- `--model_name` : 사용할 모델 이름 (timm모델 사용, 기본값 : resnet50)

### Test
```bash
python3 test.py --model_name resnet50 --file_path ./best_model.pt
```
- `--model_name` : 모델 아키텍쳐 이름 (필수)
- `--file_path` : 저장된 모델 파일 경로 (필수)

<br />

## 🏅 Wrap-Up Report   
- [Wrap-Up Report👑]
