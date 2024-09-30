import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

def plot_losses(epoch, train_losses, val_losses, result_path, model_name):
    best_model_dir = os.path.join(result_path, 'best_model')
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    })

    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='o')

    plt.title(f"{model_name} Loss Progression")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 모델 이름을 포함하여 파일 저장
    plt.savefig(os.path.join(best_model_dir, f"{model_name}_loss_plot_epoch_{epoch}.png"))
    plt.close()

def measure_time(start_time,end_time):
    t = end_time - start_time
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return f"경과 시간 = {hours}:{minutes}:{seconds}"

class EarlyStopping:
    def __init__(self, model_name, patience=7, verbose=False, delta=0, result_path='./result'):
        self.model_name = model_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.result_path = result_path

        self.top_models = []

    def __call__(self, val_loss, model, epoch):
        """검증 손실이 개선되었을 때 모델을 저장"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """성능이 좋은 상위 3개의 모델만 저장하고 나머지는 삭제"""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        checkpoint_dir = os.path.join(self.result_path, 'check_point')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_file_path = os.path.join(checkpoint_dir, f'{self.model_name}_checkpoint_{epoch}.pt')

        torch.save(model.state_dict(), model_file_path)
        print(f"Model checkpoint saved: {model_file_path}")

        self.top_models.append((val_loss, model_file_path))

        self.top_models = sorted(self.top_models, key=lambda x: x[0])

        while len(self.top_models) > 3:
            _, model_to_delete = self.top_models.pop(0)  # 가장 오래된 모델을 삭제
            if os.path.exists(model_to_delete):
                os.remove(model_to_delete)
                print(f"Model deleted: {model_to_delete}")

        self.val_loss_min = val_loss

    def save_best_model(self, epochs):
        best_model_file = self.top_models[0][1]

        best_model_dir = os.path.join(self.result_path, 'best_model')

        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        best_model_path = os.path.join(best_model_dir, f'{self.model_name}_{epochs}_best_model.pt')

        best_model_state_dict = torch.load(best_model_file)
        torch.save(best_model_state_dict, best_model_path)
        print(f"Best model saved at {best_model_path}")



