# =============================================================================
# Import required libraries
# =============================================================================
import timeit
from tqdm import tqdm
from copy import deepcopy

import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

from evaluation_metrics import EvaluationMetrics

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine():
    def __init__(self,
                 args,
                 model,
                 criterion,
                 train_loader,
                 validation_loader,
                 classes):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.classes = classes

    def learnabel_parameters(self, model):
        return [p for p in model.parameters() if p.requires_grad == True]

    def count_learnabel_parameters(self, parameters):
        return sum(p.numel() for p in parameters)

    def cycle_scheduler(self, optimizer, lr):
        steps_per_epoch = len(self.train_loader)
        return OneCycleLR(optimizer,
                          max_lr=lr,
                          steps_per_epoch=steps_per_epoch,
                          epochs=self.args.epochs,
                          pct_start=0.2)

    def initialize_optimizer(self):
        lr = self.args.learning_rate
        self.optimizer = optim.Adam(self.learnabel_parameters(self.model),
                                    lr)
        self.scheduler = self.cycle_scheduler(self.optimizer, lr)

    def initialization(self):
        if not self.args.evaluate:
            self.initialize_optimizer()
            self.best_f1_score = 0

            backbone_param = self.count_learnabel_parameters(
                self.learnabel_parameters(self.model.backbone))
            head_param = self.count_learnabel_parameters(
                self.learnabel_parameters(self.model.head))

            print('Number of Backbone\'s learnable parameters: ' +
                  str(backbone_param))
            print('Number of Head\'s learnable parameters: ' + str(head_param))
            #
            print('Optimizer: {}'.format(self.optimizer))

        self.metrics = EvaluationMetrics(self.args)

        if not torch.cuda.is_available():
            print('CUDA is not available. Training on CPU ...')
        else:
            print('CUDA is available! Training on GPU ...')
            print(torch.cuda.get_device_properties('cuda'))
        #
        self.model.to(device)
        #
        if self.args.data == 'VG-500':
            self.model_ema = ModelEmaV2(self.model, 0.999, device=device)
        else:
            self.model_ema = ModelEmaV2(self.model, 0.99, device=device)

    def print_metrics(self, results):
        N_plus = 'N+: {:.0f}'.format(results['N+'])
        per_class_metrics = 'per-class precision: {:.4f} \t per-class recall: {:.4f} \t per-class f1: {:.4f}'.format(
            results['per_class/precision'], results['per_class/recall'], results['per_class/f1'])
        m_ap = 'm_AP: {:.4f}'.format(results['m_ap'])
        return N_plus, per_class_metrics, m_ap

    def print_ema_metrics(self, results):
        ema_N_plus = 'ema_N+: {:.0f}'.format(results['ema_N+'])
        ema_per_class_metrics = 'ema_per-class precision: {:.4f} \t ema_per-class recall: {:.4f} \t ema_per-class f1: {:.4f}'.format(
            results['ema_per_class/precision'], results['ema_per_class/recall'], results['ema_per_class/f1'])
        ema_m_ap = 'ema_m_AP: {:.4f}'.format(results['ema_m_ap'])
        return ema_N_plus, ema_per_class_metrics, ema_m_ap

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model.path))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model.path)

    def mix_up_images(self, image_data, epoch):
        images = torch.zeros_like(image_data)
        batch_size = image_data.shape[0]
        half = batch_size // 2
        images[half:] = image_data[half:]
        images[:half] = (image_data[:half] + image_data[half:]) / 2
        return images

    def mix_up_labels(self, target_data, epoch):
        targets = torch.zeros_like(target_data)
        batch_size = target_data.shape[0]
        half = batch_size // 2
        targets[half:] = target_data[half:]
        targets[:half] = target_data[:half] + target_data[half:]
        # check if duplicate happened
        targets[targets > 1] = 1
        return targets

    def train(self, dataloader, epoch=None, threshold=0.5):
        train_loss = 0
        total_outputs = []
        total_targets = []
        self.model.train()

        for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            targets = targets.to(device)

            if self.args.mixup and images.shape[0] % 2 == 0:
                images = self.mix_up_images(images, epoch)
                targets = self.mix_up_labels(targets, epoch)

            # zero the gradients parameter
            self.optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to
            # the model
            outputs, _ = self.model(images)

            # calculate the batch loss
            loss = self.criterion(outputs, targets)

            # backward pass: compute gradient of the loss with respect to
            # the model parameters
            loss.backward()

            # parameters update
            self.optimizer.step()

            # learning rate update
            self.scheduler.step()

            #
            self.model_ema.update(self.model)

            train_loss += loss.item()
            total_outputs.append(torch.sigmoid(outputs))
            total_targets.append(targets)

        results = self.metrics.calculate_metrics(
            torch.cat(total_targets),
            torch.cat(total_outputs),
            threshold=threshold)

        print('Epoch: {}'.format(epoch+1))
        print('Train Loss: {:.5f}'.format(train_loss/(batch_idx+1)))
        #
        N_plus, per_class_metrics, m_ap = self.print_metrics(
            results)
        print(N_plus)
        print(per_class_metrics)
        print(m_ap)

    def validation(self, dataloader, epoch=None, threshold=0.5):
        valid_loss = 0
        total_outputs = []
        if not self.args.evaluate:
            total_outputs_ema = []
        total_targets = []
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):

                images = images.to(device)
                targets = targets.to(device)

                outputs, _ = self.model(images)
                if not self.args.evaluate:
                    outputs_ema, _ = self.model_ema.module(images)

                loss = self.criterion(outputs, targets)
                valid_loss += loss.item()

                total_outputs.append(torch.sigmoid(outputs))
                if not self.args.evaluate:
                    total_outputs_ema.append(torch.sigmoid(outputs_ema))
                total_targets.append(targets)

        results = self.metrics.calculate_metrics(
            torch.cat(total_targets),
            torch.cat(total_outputs),
            torch.cat(total_outputs_ema) if not self.args.evaluate else None,
            threshold=threshold)

        print('Validation Loss: {:.5f}'.format(valid_loss/(batch_idx+1)))
        #
        N_plus, per_class_metrics, m_ap = self.print_metrics(
            results)
        print(N_plus)
        print(per_class_metrics)
        print(m_ap)

        # save model when 'per-class f1-score' of the validation set improved
        if not self.args.evaluate:
            # print EMA
            ema_N_plus, ema_per_class_metrics, ema_m_ap = self.print_ema_metrics(
                results)
            print(ema_N_plus)
            print(ema_per_class_metrics)
            print(ema_m_ap)
            #
            if results['per_class/f1'] > self.best_f1_score:
                print('per-class f1 increased ({:.4f} --> {:.4f}). saving model ...'.format(
                    self.best_f1_score, results['per_class/f1']))
                # save the model's best result on the 'checkpoints' folder
                self.save_model()
                #
                self.best_f1_score = results['per_class/f1']

    def train_iteration(self):
        print('==> Start of Training ...')
        for epoch in range(self.args.epochs):
            start = timeit.default_timer()
            self.train(self.train_loader, epoch)
            self.validation(self.validation_loader, epoch)
            print('LR {:.1e}'.format(self.scheduler.get_last_lr()[0]))
            stop = timeit.default_timer()
            print('time: {:.3f}'.format(stop - start))
        print('==> End of training ...')


# Exponential Moving Average (EMA)
class ModelEmaV2(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e,
                     m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
