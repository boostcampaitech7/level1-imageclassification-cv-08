import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def inference(model, device, test_loader):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader,desc='Inference'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def inference_preds(model, device, test_loader):
    model.to(device)
    model.eval()
    predictions = None  # 배열로 초기화
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Inference'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()  # 확률 계산 후 numpy로 변환
            
            # 배열 초기화 및 누적
            if predictions is None:
                predictions = probs  # 첫 배치에서는 초기화
            else:
                predictions = np.vstack((predictions, probs))  # 기존 배열에 세로로 쌓기
    return predictions