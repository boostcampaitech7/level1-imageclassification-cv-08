import torch
from tqdm import tqdm
import os
from utils.utils import plot_losses
from configs.config import Config
from torch.cuda.amp import autocast, GradScaler
from utils.utils import EarlyStopping
from trainers.test_runner import TestRunner

class Trainer:
    def __init__(self, model, device,
                 train_loader, val_loader, scheduler,
                 optimizer, loss_fn, epochs, result_path, patience=7):
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

        self.early_stopping = EarlyStopping(patience = patience,
                                            verbose = True,
                                            result_path = result_path)

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
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.val_loader)

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

            self.early_stopping(v_loss, self.model, self.epochs)

            if self.early_stopping.early_stop:
                print('Early Stopping Training stopped')
                break

            if config.scheduler.what_scheduler == 'reduce':
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

        plot_losses(self.epochs,
                    self.train_losses, 
                    self.val_losses,
                    self.result_path)
        
        self.run_test()

    def run_test(self):
        test_runner = TestRunner(self.model, Config())
        test_runner.load_model()
        test_runner.run_test()

