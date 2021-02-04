import os
import threading

import joblib
import six
import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
from torchsummary import summary

from utils import utils

logger = utils.get_logger('Trainer')


class RS3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, skip_train_validation=False,
                 multi_head=False, dist_t=True, use_amp=False):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.baseline = multi_head
        self.use_amp = use_amp
        logger.info(model)
        logger.info(
            f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(
            log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter
        self.dis_transform = dist_t
        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation
        #self.clf = joblib.load('../radiomics_work/trained.pkl')

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, skip_train_validation=False, multi_head=False, dist_t=True, epoches=None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=epoches,
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   tensorboard_formatter=tensorboard_formatter,
                   skip_train_validation=skip_train_validation,
                   multi_head=multi_head,
                   dist_t=dist_t
                   )

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, skip_train_validation=False, multi_head=False, dist_t=True):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   tensorboard_formatter=tensorboard_formatter,
                   skip_train_validation=skip_train_validation,
                   multi_head=multi_head, dist_t=dist_t)

    def fit(self):

        self.scaler = GradScaler()

        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                logger.info(
                    'Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1
        logger.info(
            f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """

        # initialize statistic recorders
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()
        # clear the gradients
        self.optimizer.zero_grad()

        # training loop
        for i, t in enumerate(train_loader):

            # prepare input & target
            input, target, _ = self._split_training_batch(t)
            target = target.cuda()

            S0, I1, S1, I2, S2, S3 = 0, 0, 0, 0, 0, 0

            def FORWARD():
                # forward pass
                Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3 = self._forward_pass(
                    input, target)
                loss = Ls1 + Ls2 + Lr1 + Lr2 + Li
                # summary for multiple GPUs
                Ls1 = Ls1.sum()
                Ls2 = Ls2.sum()
                Lr1 = Lr1.sum()
                Lr2 = Lr2.sum()
                Li = Li.sum()
                loss = loss.sum()

                # calc evaluation score
                eval_score = self.eval_criterion(S3, target)

                # log loss & evaluation statistics
                train_losses.update(
                    loss.mean().item(), self._batch_size(input))
                train_eval_scores.update(
                    eval_score.mean().item(), self._batch_size(input))
                self._log_stats(
                    'train', loss, eval_score, Ls1, Ls2, Lr1, Lr2, Li
                )
                return loss, S0, I1, S1, I2, S2, S3

            # automatically adjust precision
            if self.use_amp:
                with autocast():
                    loss, S0, I1, S1, I2, S2, S3 = FORWARD()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, S0, I1, S1, I2, S2, S3 = FORWARD()
                loss.backward()
                self.optimizer.step()

            # validate model every [validate_after_iters] steps
            if self.num_iterations % self.validate_after_iters == 0:

                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            # tensorboard log parameters and images
            if self.num_iterations % self.log_after_iters == 0:
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. \
                        Evaluation score: {train_eval_scores.avg}'
                )
                self._log_params()
                self._log_images(
                    input, target, S0, I1, S1, I2, S2, S3,
                    prefix='train_', num_iters=self.num_iterations
                )

            # stop judgement
            if self.should_stop():
                return True

            # add count
            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(
                f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self, val_loader):

        # logger
        logger.info('Validating...')

        # initialize statistics recorders
        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        vals1_losses = utils.RunningAverage()
        vals2_losses = utils.RunningAverage()
        valr1_losses = utils.RunningAverage()
        valr2_losses = utils.RunningAverage()
        vali_losses = utils.RunningAverage()

        # evaluation epoch
        with torch.no_grad():
            for i, t in enumerate(val_loader):

                #logger.info(f'Validation iteration {i}')

                # prepare input & target
                input, target, _ = self._split_training_batch(t)
                target = target.cuda()

                # forward pass
                Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3 = self._forward_pass(
                    input, target)
                loss = Ls1 + Ls2 + Lr1 + Lr2 + Li

                # log loss & evaluation statistics
                val_losses.update(loss.mean().item(), self._batch_size(input))
                vals1_losses.update(Ls1.mean().item(), self._batch_size(input))
                vals2_losses.update(Ls2.mean().item(), self._batch_size(input))
                valr1_losses.update(Lr1.mean().item(), self._batch_size(input))
                valr2_losses.update(Lr2.mean().item(), self._batch_size(input))
                vali_losses.update(Li.mean().item(), self._batch_size(input))

                # calc evaluation score
                eval_score = self.eval_criterion(S3, target)
                val_scores.update(eval_score.mean().item(),
                                  self._batch_size(input))

                # log images
                if i % self.log_after_iters == 0:
                    self._log_images(
                        input, target, S0, I1, S1, I2, S2, S3, num_iters=i, prefix='val_'
                    )

                # stop validation
                if self.validate_iters is not None and self.validate_iters <= i:
                    break

            # log stats
            self._log_stats(
                'val', val_losses.avg, val_scores.avg, vals1_losses.avg,
                vals2_losses.avg, valr1_losses.avg, valr2_losses.avg, vali_losses.avg
            )
            logger.info(
                f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}'
            )
            return val_scores.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        cls = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, cls = t
        if self.dis_transform == False:
            return input, target*1.0, cls
        else:
            return input, target*1.0, cls

    def clf_loss(self, r_features, cls_gt, p):
        loss = torch.zeros(1).cuda()
        for b in range(r_features.shape[0]):
            for pre_mask in range(r_features.shape[1]):
                this_one = r_features[b:b+1, pre_mask, :]
                pred = self.clf.predict(this_one)*1.0
                pred = torch.tensor(
                    [1-pred, pred]).cuda().reshape((cls_gt.shape[0], -1)).float()
                loss += torch.nn.functional.cross_entropy(
                    pred, cls_gt)*p[b, pre_mask]

        return loss/p.sum()

    def _forward_pass(self, input, target):
        # forward pass
        Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3 = self.model(
            input, target, self.baseline)
        return Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,

        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg, s1, s2, r1, r2, i):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg,
            f'{phase}_loss_reg1': s1,
            f'{phase}_loss_seg1': s2,
            f'{phase}_loss_reg2': r1,
            f'{phase}_loss_seg2': r2,
            f'{phase}_loss_imp': i,
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            try:
                self.writer.add_histogram(
                    name, value.data.cpu().numpy(), self.num_iterations)
                self.writer.add_histogram(
                    name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)
            except:
                continue

    def _normalize(self, pic):
        _min = pic.min()
        _max = pic.max()
        return (pic - _min) / (_max - _min)

    def _log_images(self, input, segs, seg0, reg1, seg1, reg2, seg2, seg3, num_iters, prefix=''):
        """
        segs = (segs >= 0.)*1.0
        seg0 = (seg0 >= 0.) * 1.0
        seg1 = (seg1 >= 0.) * 1.0
        seg2 = (seg2 >= 0.) * 1.0
        seg3 = (seg3 >= 0.) * 1.0
        """
        F = torch.nn.functional
        segs = F.normalize(segs)
        seg0 = F.normalize(seg0)
        seg1 = F.normalize(seg1)
        seg2 = F.normalize(seg2)
        seg3 = F.normalize(seg3)

        inputs_map = {
            'inputs': input,
            'target': segs,
            'pred_seg': seg0,
            'reg_inputs': reg1,
            'reg_seg': seg1,
            'seg2': seg2,
            'reg_reg_input': reg2,
            'reg_seg2': seg3,
        }

        img_sources = {}
        for name, batch in inputs_map.items():
            img_sources[name] = batch.data.cpu().numpy()

        # find the slice which has biggest area in the volume. [N]
        idx_slice = torch.argmax(segs.sum(-1).sum(-1).sum(1), -1).byte()
        for i, (name, batch) in enumerate(img_sources.items()):
            for tag, image in self.tensorboard_formatter(name, batch, idx_slice, num_iters):
                self.writer.add_image(
                    prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
