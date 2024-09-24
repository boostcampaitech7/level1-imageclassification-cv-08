import torch
import torch.nn.functional as F
from tqdm import tqdm

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