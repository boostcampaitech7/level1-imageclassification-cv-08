from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(outputs, targets, average='macro'):
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)

    return precision, recall, f1
