import os
import threading

import joblib
import numpy as np
import six
import cv2
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from utils import utils

#from radiomics.featureextractor import RadiomicsFeatureExtractor

logger = utils.get_logger('UNet3DTrainer')
#extractor = RadiomicsFeatureExtractor('../radiomics_work/RadiomicsParams.yaml')


class MyThread(threading.Thread):
    def __init__(self, a, b, ):
        threading.Thread.__init__(self)
        #self.name = name
        self.extractor = RadiomicsFeatureExtractor(
            '../radiomics_work/RadiomicsParams.yaml')
        self.img = a
        self.mask = b

    def run(self):
        try:
            print('start!')
            self.result = self.extractor.execute(self.img, self.mask)
            print('finish!')
        except:
            print('Ex stop')

    def get_result(self):
        try:
            print('yes!')
            return self.result
        except Exception:
            return None


class UNet3DTrainer:
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
                 tensorboard_formatter=None, skip_train_validation=False, multi_head=True):

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
        self.multi_head = multi_head
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

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation
        #self.clf = joblib.load('../radiomics_work/trained.pkl')

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, skip_train_validation=False, multi_head=False):
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
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   tensorboard_formatter=tensorboard_formatter,
                   skip_train_validation=skip_train_validation,
                   multi_head=multi_head
                   )

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, skip_train_validation=False, multi_head=False):
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
                   multi_head=multi_head)

    def fit(self):
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
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            # logger.info(
            #    f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, target, cls = self._split_training_batch(t)
            if target.shape[1] > 1:
                ttemp = []
                for ii in range(target.shape[1]):
                    ttemp.append(torch.zeros(target.shape[0], 2, target.shape[2],
                                             target.shape[3], target.shape[4]).scatter_(1, target[:, ii:ii+1, :, :, :].cpu().long(), 1))
                target = torch.cat(ttemp, 1)
            else:
                target = torch.zeros(target.shape[0], 2, target.shape[2],
                                     target.shape[3], target.shape[4]).scatter_(1, target.cpu().long(), 1)
            if target.max() == 0:
                a = 1
            target = target.cuda()
            output, loss, p = self._forward_pass(input, target, cls)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                print('1')
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

            if self.num_iterations % self.log_after_iters == 0:

                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                # compute eval criterion
                if not self.skip_train_validation:
                    if self.multi_head:
                        eval_score = torch.zeros(1).cuda()
                        for j in range(self.head_num):
                            eval_score += self.eval_criterion(
                                output[:, j*2:j*2+2, :, :, :], target)
                        eval_score /= self.head_num
                    else:
                        if target.shape[1] > 2:
                            eval_score = torch.zeros(1).cuda()
                            for j in range(target.shape[1]//2):
                                eval_score += self.eval_criterion(output[:, j * 2:j * 2 + 2, :, :, :],
                                                                  target[:, j * 2:j * 2 + 2, :, :, :])
                            eval_score /= target.shape[1]//2
                        else:
                            eval_score = self.eval_criterion(output, target)
                    train_eval_scores.update(
                        eval_score.item(), self._batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg,
                                train_eval_scores.avg)
                self._log_params()
                self._log_images(input, target, output, 'train_')
                a = 1
            if self.should_stop():
                return True

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
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(val_loader):
                logger.info(f'Validation iteration {i}')

                input, target, weight = self._split_training_batch(t)
                if target.shape[1] > 1:
                    ttemp = []
                    for ii in range(target.shape[1]):
                        ttemp.append(torch.zeros(target.shape[0], 2, target.shape[2],
                                                 target.shape[3], target.shape[4]).scatter_(1,
                                                                                            target[:, ii:ii + 1, :, :,
                                                                                                   :].cpu().long(), 1))
                    target = torch.cat(ttemp, 1)
                else:
                    target = torch.zeros(target.shape[0], 2, target.shape[2],
                                         target.shape[3], target.shape[4]).scatter_(1, target.cpu().long(), 1)
                target = target.cuda()
                output, loss, p = self._forward_pass(input, target, weight)
                val_losses.update(loss.item(), self._batch_size(input))

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                if i % 100 == 0:
                    self._log_images(input, target, output, 'val_')

                if self.multi_head:
                    eval_score = torch.zeros(1).cuda()
                    for j in range(self.head_num):
                        eval_score += p[:, j]*self.eval_criterion(
                            output[:, j * 2:j * 2 + 2, :, :, :], target)
                    eval_score /= p.sum()
                else:
                    eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg)
            logger.info(
                f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
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
        return input, target, cls

    def get_margin(self, segs):
        #segs = sitk.ReadImage(segs)
        segs_ar = sitk.GetArrayFromImage(segs)

        kernel = np.ones((3, 3), np.uint8)
        segs_ar_margin = cv2.dilate(
            segs_ar, kernel, iterations=1) - cv2.erode(segs_ar, kernel, iterations=1)
        segs_margin = sitk.GetImageFromArray(segs_ar_margin)
        segs_margin.CopyInformation(segs)
        return segs_margin

    def r_thread(self, pred_mask, input_img):
        print('1st t')
        v = []
        tt = []
        mask = (pred_mask > 0.5).astype(np.uint8)
        mask = sitk.GetImageFromArray(mask)
        for mod in range(6):
            img = sitk.GetImageFromArray(input_img[mod, :, :, :])
            t = MyThread(self.second_thread, img, mask)
            tt.append(t)
        for mod in range(6):
            tt[mod].start()
        for mod in range(6):
            tt[mod].join()
        for mod in range(6):
            result = tt[mod].get_result()
            for jj, (key, val) in enumerate(six.iteritems(result)):
                if jj < 11:
                    continue
                if not isinstance(val, (float, int, np.ndarray)):
                    continue
                if np.isnan(val):
                    val = 0
                # print(key)
                v.append(val)
            #result = extractor.execute(img, self.get_margin(mask))
            # for jj, (key, val) in enumerate(six.iteritems(result)):
            #    if jj < 11:
            #        continue
            #    if not isinstance(val, (float, int, np.ndarray)):
            #        continue
            #    if np.isnan(val):
            #        val = 0
                # print(val)
               # v.append(val)
        return v

    def get_r(self, pred_mask, input_img):
        pred_mask = pred_mask.sigmoid().detach().cpu().numpy()
        input_img = input_img.cpu().numpy()
        All_f = []
        for j in range(pred_mask.shape[0]):
            Fea = []
            tt = []
            for i in range(0, self.head_num*2, 2):
                mask = (pred_mask[j, i, :, :] > 0.5).astype(np.uint8)
                mask = sitk.GetImageFromArray(mask)
                for mod in range(6):
                    img = sitk.GetImageFromArray(input_img[j, mod, :, :, :])
                    t = MyThread(img, mask)
                    t.start()
                    tt.append(t)

            for i in range(self.head_num):
                for mod in range(6):
                    tt[i*6+mod].join()
            for i in range(self.head_num):
                v = []
                for mod in range(6):
                    result = tt[i*6+mod].get_result()
                    for jj, (key, val) in enumerate(six.iteritems(result)):
                        if jj < 11:
                            continue
                        if not isinstance(val, (float, int, np.ndarray)):
                            continue
                        if np.isnan(val):
                            val = 0
                        # print(key)
                        v.append(val)
                Fea.append(v)
            All_f.append(Fea)
        All_f = np.array(All_f)
        return All_f

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

    def _forward_pass(self, input, target, cls=None):
        # forward pass

        if self.multi_head:
            output, p = self.model(input)
            self.head_num = 10
            # compute the loss
            loss = torch.zeros(1).cuda()
            for i in range(self.head_num):
                loss += p[:, i] * \
                    self.loss_criterion(output[:, 2*i:2*i+2, :, :, :], target,)

            # R_features=self.get_r(output,input)
            # loss2=self.clf_loss(R_features,cls,p)
            loss = loss/p.sum(1)
        else:
            output = self.model(input)
            # compute the loss
            ndd = target.shape[1]//2
            # loss=torch.zeros(1).cuda()
            loss = self.loss_criterion(output, target)
            # for i in range(ndd):
            #     loss += self.loss_criterion(output[:,i*2:i*2+1,:,:,:], target[:,i*2:i*2+1,:,:,:])
            p = 0
        return output, loss, p

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
        if self.multi_head:
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
                'mu': self.model.mu,
                'bias': self.model.bias,
            }, is_best, checkpoint_dir=self.checkpoint_dir,
                logger=logger)
        else:
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

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(
                name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(
                name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(
                    prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)


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
                 multi_head=False, dist_t=True):

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
        train_losses = utils.RunningAverage()
        trains1_losses = utils.RunningAverage()
        trains2_losses = utils.RunningAverage()
        trainr1_losses = utils.RunningAverage()
        trainr2_losses = utils.RunningAverage()
        traini_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()
        #eval_score = self.validate(self.loaders['val'])

        # sets the model in training mode
        self.model.train()
        #val_score = self.validate(self.loaders['val'])
        # print(len(train_loader))
        self.optimizer.zero_grad()
        for i, t in enumerate(train_loader):
            # logger.info(
            #    f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
            # print(self.num_iterations)
            input, target, _ = self._split_training_batch(t)

            target = target.cuda()
            Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3 = self._forward_pass(
                input, target)
            loss = Ls1 + Ls2 + Lr1 + Lr2 + Li
            #self._log_images(input, target, I1, S1, I2, S2, S3, 'train_')
            train_losses.update(loss.item(), self._batch_size(input))
            trains1_losses.update(Ls1.item(), self._batch_size(input))
            trains2_losses.update(Ls2.item(), self._batch_size(input))
            trainr1_losses.update(Lr1.item(), self._batch_size(input))
            trainr2_losses.update(Lr2.item(), self._batch_size(input))
            traini_losses.update(Li.item(), self._batch_size(input))
            # compute gradients and update parameters
            eval_score = self.eval_criterion(S3, target)
            train_eval_scores.update(
                eval_score.item(), self._batch_size(input))
            self._log_stats('train', loss, eval_score, Ls1, Ls2, Lr1, Lr2, Li)
            loss.backward()

            if self.num_iterations % 1 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._log_stats('train', train_losses.avg, train_eval_scores.avg, trains1_losses.avg,
                                trains2_losses.avg, trainr1_losses.avg, trainr2_losses.avg, traini_losses.avg)
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
                # compute eval criterion
            if self.num_iterations % self.log_after_iters == 0:
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_params()
                self._log_images(input, target, S0, I1,
                                 S1, I2, S2, S3, 'train_')
            if self.should_stop():
                return True
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
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        vals1_losses = utils.RunningAverage()
        vals2_losses = utils.RunningAverage()
        valr1_losses = utils.RunningAverage()
        valr2_losses = utils.RunningAverage()
        vali_losses = utils.RunningAverage()
        with torch.no_grad():
            for i, t in enumerate(val_loader):
                #logger.info(f'Validation iteration {i}')

                input, target, _ = self._split_training_batch(t)
                target = target.cuda()
                Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3 = self._forward_pass(
                    input, target)
                loss = Ls1+Ls2+Lr1+Lr2+Li
                val_losses.update(loss.item(), self._batch_size(input))
                vals1_losses.update(Ls1.item(), self._batch_size(input))
                vals2_losses.update(Ls2.item(), self._batch_size(input))
                valr1_losses.update(Lr1.item(), self._batch_size(input))
                valr2_losses.update(Lr2.item(), self._batch_size(input))
                vali_losses.update(Li.item(), self._batch_size(input))
                if i % 100 == 0:
                    self._log_images(input, target, S0, I1,
                                     S1, I2, S2, S3, 'val_')

                eval_score = self.eval_criterion(S3, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg, vals1_losses.avg,
                            vals2_losses.avg, valr1_losses.avg, valr2_losses.avg, vali_losses.avg)
            logger.info(
                f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
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
        # compute the loss
        # loss=Ls1+Ls2+Lr1+Lr2+Li
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

    def _log_images(self, input, segs, seg0, reg1, seg1, reg2, seg2, seg3, prefix=''):
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
        # self._log_images(input, target, S0, I1, S1, I2, S2, S3, 'train_')
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()
        idx_slice = torch.argmax(segs.sum(-1).sum(-1).sum(1), -1)
        for i, (name, batch) in enumerate(img_sources.items()):
            for tag, image in self.tensorboard_formatter(name, batch, idx_slice):
                self.writer.add_image(
                    prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
