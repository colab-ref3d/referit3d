"""
Utilities to analyze, train, test an 3d_listener.
"""

import torch
import numpy as np
import pandas as pd
import tqdm
import torch.nn.functional as F

from ..utils.evaluation import DistAverageMeter as AverageMeter


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    if args.cl_alpha > 0:
        batch_keys.append('target_class_mask')

    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, dist_mgr, args):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param dist_mgr: distributed training manager
    :type dist_mgr: BaseDistMgr
    :param args:
    :return:
    """
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter(dist_mgr)
    referential_loss_mtr = AverageMeter(dist_mgr)
    cl_loss_mtr = AverageMeter(dist_mgr)
    obj_loss_mtr = AverageMeter(dist_mgr)
    ref_acc_mtr = AverageMeter(dist_mgr)
    cls_acc_mtr = AverageMeter(dist_mgr)
    txt_acc_mtr = AverageMeter(dist_mgr)

    # Set the model in training mode
    model.train()
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)

    iterator = data_loader
    if dist_mgr.get_rank() == 0:
        iterator = tqdm.tqdm(data_loader)
    for batch in iterator:
        # Move data to gpu
        for k in batch_keys:
            batch[k] = batch[k].to(device, non_blocking=True)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        # Backward
        optimizer.zero_grad()
        all_losses = compute_losses(batch, res, criteria, args)
        total_loss = all_losses['total_loss']
        avg_loss = total_loss / dist_mgr.get_world_size()
        avg_loss.backward()
        model.sync_gradients()  # dist training
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(total_loss.item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        # TODO copy the ref-loss to homogeneize the code
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

        if args.cl_alpha > 0:
            cl_loss_mtr.update(all_losses['cl_loss'], batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_loss'] = referential_loss_mtr.avg
    metrics['train_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    metrics['train_cl_loss'] = cl_loss_mtr.avg
    return metrics


def compute_losses(batch, res, criterion_dict, args):
    """Calculate the loss given the model logits and the criterion
    :param batch:
    :param res: dict of logits
    :param criterion_dict: dict of the criterion should have key names same as the logits
    :param args, argparse.Namespace
    :return: scalar loss value
    """

    obj_clf_loss = lang_clf_loss = referential_loss = cl_loss = 0.

    total_loss = 0.

    # Get the object language classification loss and the object classification loss
    if args.ref_cls_alpha > 0:
        criterion = criterion_dict['logits']
        logits = res['logits']
        if args.s_vs_n_weight is not None:
            referential_loss = criterion(logits, batch)
        else:
            referential_loss = criterion(logits, batch['target_pos'])
        total_loss += referential_loss * args.ref_cls_alpha

    if args.obj_cls_alpha > 0:
        criterion = criterion_dict['class_logits']
        obj_clf_loss = criterion(res['class_logits'].transpose(2, 1), batch['class_labels'])
        total_loss += obj_clf_loss * args.obj_cls_alpha

    if args.lang_cls_alpha > 0:
        criterion = criterion_dict['lang_logits']
        lang_clf_loss = criterion(res['lang_logits'], batch['target_class'])
        total_loss += lang_clf_loss * args.lang_cls_alpha

    if args.cl_alpha > 0:
        graph_features = res['graph_features']
        language_features = res['language_features']  # TODO hard linear?
        target_pos = batch['target_pos']
        target_msk = batch['target_class_mask']  # B x Nobj

        def _dot(a, b):
            return torch.sum(a * b, -1)
        language_features = language_features.unsqueeze(1)  # B x 1 x Ndim

        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.logits = logits
                self.reduce = reduce

            def forward(self, inputs, targets):
                if self.logits:
                    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
                else:
                    BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
                pt = torch.exp(-BCE_loss)
                F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

                if self.reduce:
                    return torch.mean(F_loss)
                else:
                    return F_loss

        if args.cl_type == 'infonce':
            sim = _dot(language_features, graph_features)  # B x Nobj
            expsim = sim.exp() * target_msk  # set non-distractors' expsim to zero
            cl_logits = expsim / expsim.sum(-1, keepdim=True)
            cl_logits[target_msk.byte()].log_()  # log-softmax
            cl_loss = torch.nn.NLLLoss()(cl_logits, target_pos)
        elif args.cl_type == 'dim':
            assert 'dim_logits' in res, 'for dim learning, should output the sim score'
            dim = res['dim_logits']
            onehot = torch.zeros_like(dim)
            onehot.scatter_(1, target_pos, 1)
            # using focal loss to balance the pos-neg samples
            cl_loss = FocalLoss()(dim, onehot)

        total_loss += cl_loss * args.cl_alpha


    return {'total_loss': total_loss, 'referential_loss': referential_loss,
            'obj_clf_loss': obj_clf_loss, 'lang_clf_loss': lang_clf_loss,
            'cl_loss': cl_loss,
            }


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, dist_mgr, args, randomize=False):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter(dist_mgr)
    referential_loss_mtr = AverageMeter(dist_mgr)
    obj_loss_mtr = AverageMeter(dist_mgr)
    cl_loss_mtr = AverageMeter(dist_mgr)
    ref_acc_mtr = AverageMeter(dist_mgr)
    cls_acc_mtr = AverageMeter(dist_mgr)
    txt_acc_mtr = AverageMeter(dist_mgr)

    # Set the model in training mode
    model.eval()

    if randomize:
        np.random.seed()  # call this to change the sampling of the point-clouds #TODO-A talk about it.
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)

    iterator = data_loader
    if dist_mgr.get_rank() == 0:
        iterator = tqdm.tqdm(data_loader)

    for batch in iterator:
        # Move data to gpu
        for k in batch_keys:
            batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        all_losses = compute_losses(batch, res, criteria, args)

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(all_losses['total_loss'].item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)
        if args.cl_alpha > 0:
            cl_loss_mtr.update(all_losses['cl_loss'], batch_size)

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_loss'] = referential_loss_mtr.avg
    metrics['test_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    metrics['test_cl_loss'] = cl_loss_mtr.avg
    return metrics


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        out = model(batch)

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])

        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])
    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=2020):
    """
    Return the predictions along with the scan data for further visualization
    """
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)

    for batch in data_loader:
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass
        res = model(batch)

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'is_easy': batch['is_easy'][i]
            })

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
