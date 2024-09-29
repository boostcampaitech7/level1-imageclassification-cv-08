import torch
from tqdm import tqdm
from utils.utils import EarlyStopping,plot_losses
from data.cutmix_loader import CutMixLoader
from trainers.metric import compute_metrics

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, config, alpha=1.0):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.optimizer = optimizer
        self.alpha = alpha

        self.early_stopping = EarlyStopping(model_name=self.config["model"]["model_name"], patience=config['training']['early_stop_patience'], result_path=config['result_path'], verbose=True)
        self.scaler = torch.amp.GradScaler('cuda')

    def train_basic(self):
        """기본 학습을 수행하는 함수"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def train_cutmix(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(CutMixLoader(self.train_loader, self.alpha), desc='Training (CutMix)', leave=False)

        for images, targets_a, targets_b, lam in progress_bar:
            images, targets_a, targets_b = images.to(self.device), targets_a.to(self.device), targets_b.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast():
                outputs = self.model(images)
                loss = lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_targets, all_outputs = [],[]
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

                all_targets.append(targets)
                all_outputs.append(outputs)
        all_targets, all_outputs = torch.cat(all_targets),torch.cat(all_outputs)
        precision, recall, f1 = compute_metrics(all_outputs, all_targets)
        print(f'Validation Precision : {precision:.3f}, Recall : {recall:.3f}, f1 score : {f1:.3f}')
        return total_loss / len(self.val_loader), precision, recall, f1

    def train(self, use_cutmix=False):
        train_losses = []
        val_losses = []
        all_pre, all_re, all_f1 = [],[],[]

        for epoch in range(self.config['training']['epochs']):
            if use_cutmix:
                train_loss = self.train_cutmix()
            else:
                train_loss = self.train_basic()

            val_loss,precision,recall,f1 = self.validate_epoch()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            all_pre.append(precision)
            all_re.append(recall)
            all_f1.append(f1)

            print(f'Epoch [{epoch+1}/{self.config["training"]["epochs"]}], Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')

            self.early_stopping(val_loss, self.model, epoch)

            if self.early_stopping.early_stop:
                print("조기 종료되었습니다.")
                break

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
        print('[최종 결과]')
        print(f'Final Precision : {precision:.3f}, Recall : {recall:.3f}, f1 score : {f1:.3f}')
        self.early_stopping.save_best_model(self.config['training']['epochs'])
        plot_losses(self.config['training']['epochs'], train_losses, val_losses, self.config['result_path'], self.config['model']['model_name'])
        print("학습 완료")

