import torch
from tqdm import tqdm
import os
from utils.utils import plot_losses
from configs.config import Config
from torch.cuda.amp import autocast, GradScaler
from utils.utils import EarlyStopping
from trainers.test_runner import TestRunner
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer:
    def __init__(self, model, device,
                 train_loader, val_loader, scheduler,
                 optimizer, loss_fn, epochs, result_path, patience=7, gradient_accumulation_step = 1):
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
        self.gradient_accumulation_step = gradient_accumulation_step

        self.early_stopping = EarlyStopping(patience = patience,
                                            verbose = True,
                                            result_path = result_path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)

        for i, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            

            # mixed precision Training
            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()

            if (i + 1) % self.gradient_accumulation_step == 0:  # 지정한 누적 단계에 도달하면
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        
             # 마지막 남은 기울기 처리 (에포크 끝나기 직전에 남은 경우)
            if (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()

            # ACC
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader), all_targets, all_predictions
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # ACC
                _, predicted = torch.max(outputs, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader), all_targets, all_predictions

    def train(self):
        config = Config()
        final_train_targets, final_train_predictions = [], []
        final_val_targets, final_val_predictions = [], []

        for epoch in range(self.epochs):
            t_loss, train_targets, train_predictions = self.train_epoch()
            v_loss, val_targets, val_predictions = self.validate()

            final_train_targets = train_targets
            final_train_predictions = train_predictions
            final_val_targets = val_targets
            final_val_predictions = val_predictions

            # 메모리 캐시 해제
            torch.cuda.empty_cache()

            print(f'epoch {epoch+1}, Train loss: {t_loss:.5f}, Val loss: {v_loss:.5f}')
            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)

            self.early_stopping(v_loss, self.model, self.epochs)

            if self.early_stopping.early_stop:
                print('Early Stopping Training stopped')
                break

            if config.scheduler.what_scheduler == 'Reduce':
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

        final_train_accuracy = sum([1 for t, p in zip(final_train_targets, final_train_predictions) if t == p]) / len(final_train_targets)
        final_val_accuracy = sum([1 for t, p in zip(final_val_targets, final_val_predictions) if t == p]) / len(final_val_targets)

        train_precision = precision_score(final_train_targets, final_train_predictions, zero_division=0, average='macro')
        train_recall = recall_score(final_train_targets, final_train_predictions, average='macro')
        train_f1 = f1_score(final_train_targets, final_train_predictions, average='macro')

        val_precision = precision_score(final_val_targets, final_val_predictions, zero_division=0, average='macro')
        val_recall = recall_score(final_val_targets, final_val_predictions, average='macro')
        val_f1 = f1_score(final_val_targets, final_val_predictions, average='macro')

        # Print final evaluation metrics
        print(f"Final Train Accuracy: {final_train_accuracy*100:.3f}%")
        print(f"Final Train Precision: {train_precision:.3f}, Recall: {train_recall:.3f}, F1 Score: {train_f1:.3f}")
        print(f"Final Validation Accuracy: {final_val_accuracy*100:.3f}%")
        print(f"Final Validation Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1 Score: {val_f1:.3f}")

        plot_losses(self.epochs,
                    self.train_losses, 
                    self.val_losses,
                    self.result_path)
        
        self.run_test()

    def run_test(self):
        test_runner = TestRunner(self.model, Config())
        test_runner.load_model()
        test_runner.run_test()

