import pandas as pd
import matplotlib.pyplot as plt
import os
from configs.config import Config
import torch

def plot_losses(epoch,t_loss,v_loss,result_path):
    '''
    loss값 plot해주는 파트
    '''
    config = Config()
    model_name = config.model.model_name
    df = pd.DataFrame({
        'Epoch' : range(1,len(t_loss) + 1),
        'Train Loss' : t_loss,
        'Validation Loss' : v_loss
    }) 

    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='o')

    plt.title(f"{model_name}, Loss Result")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(result_path, f"{model_name}_{epoch}_Loss_plot.png"))
    plt.show()

def measure_time(start_time,end_time):
    t = end_time - start_time
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return f"경과 시간 = {hours}:{minutes}:{seconds}"

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, result_path='./train_result'):
        '''
        patience : 개선이 없는 에폭 수
        verbose : 개선시마다 메세지 출력 여부
        delta : 개선으로 간주될 최소 변화량
        result_path : 모델이 저장될 경로 (폴더로 지정)
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.result_path = result_path

        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
    
    def __call__(self, val_loss, model, epoch):
        """검증 손실을 추적하고 모델을 저장"""
        config = Config()
        model_name = config.model.model_name

        score = -val_loss

        # 첫 번째 검증 손실을 기준으로 설정
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name, epoch)

        # 검증 손실이 개선되지 않은 경우
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        # 검증 손실이 개선된 경우
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name, epoch):
        """
        검증 손실이 개선될 때 모델을 지정된 파일 이름 형식으로 저장.
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")

        model_file_path = f'{self.result_path}/{model_name}_{epoch}_best_model.pt'
        torch.save(model.state_dict(), model_file_path)

        print(f'Model saved at {model_file_path} with val loss {val_loss:.5f}')

        self.val_loss_min = val_loss