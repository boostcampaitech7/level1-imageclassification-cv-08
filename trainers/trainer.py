import torch
from tqdm import tqdm
import os
from utils.utils import plot_losses
from configs.config import Config
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, device,
                 train_loader, val_loader, scheduler,
                 optimizer, loss_fn, epochs, result_path):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.result_path = result_path
        self.best_models = []
        self.lowest_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.scaler = GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            # mixed precision Training
            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="validating", leave=False):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, epoch, loss):
        config = Config()
        os.makedirs(self.result_path, exist_ok=True)
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.3f}')

        self.best_models.append((loss,epoch, current_model_path))
        self.best_models.sort()

        if len(self.best_models) > 3:
            _,_,path_to_remove = self.best_models.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            torch.save(self.model.state_dict(),f'{self.result_path}/{config.model.model_name}_{config.training.epochs}_best_model.pt')
            print(f'모델 저장 및 loss {loss:.5f}')
            print()

    def train(self):
        config = Config()
        for epoch in range(self.epochs):
            t_loss = self.train_epoch()
            v_loss = self.validate()

            # 메모리 캐시 해제
            torch.cuda.empty_cache()

            print(f'epoch {epoch+1}, Train loss: {t_loss:.5f}, Val loss: {v_loss:.5f}')
            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)
            self.save_model(epoch, v_loss)
            if config.scheduler.what_scheduler == 'reduce':
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

        plot_losses(self.epochs,
                    self.train_losses, 
                    self.val_losses,
                    self.result_path)

