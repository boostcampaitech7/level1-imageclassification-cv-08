import pandas as pd
import matplotlib.pyplot as plt
import os
from configs.config import Config

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
