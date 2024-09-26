import pandas as pd
from collections import Counter
import os

# CSV 파일들이 있는 디렉토리 경로
csv_dir = "./twocsv"

# CSV 파일들의 경로를 불러옵니다
csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]

# 각 모델의 결과를 담을 리스트
model_predictions = []

# 각 CSV 파일에서 예측 결과를 읽어옵니다
for file in csv_files:
    df = pd.read_csv(file)
    model_predictions.append(df)

# 첫 번째 모델의 이미지 경로를 기준으로 합치기 위해 첫 번째 모델에서 image_path와 ID를 추출
image_paths = model_predictions[0]['image_path']
ids = model_predictions[0]['ID']

# 최종 다수결 결과를 담을 리스트
final_predictions = []

# 각 이미지에 대해 다수결로 예측 클래스 결정
for i, image_path in enumerate(image_paths):
    # 각 모델의 해당 이미지에 대한 예측 결과를 수집
    predictions = [df['target'].iloc[i] for df in model_predictions]
    
    # 다수결로 가장 많이 나온 클래스를 선택
    most_common_class = Counter(predictions).most_common(1)[0][0]
    
    # 최종 결과에 저장 (ID, image_path, target)
    final_predictions.append((ids.iloc[i], image_path, most_common_class))

# 최종 결과를 DataFrame으로 변환
final_df = pd.DataFrame(final_predictions, columns=['ID', 'image_path', 'target'])

# 최종 결과를 CSV 파일로 저장
final_df.to_csv("final_predictions_models_88.csv", index=False)

print("ID, image_path, target 형태로 최종 결과가 저장되었습니다.")
