import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from .protocol import VERSION, INPUT_PLANES, Plane
from .config import model_identity
from .network import InferenceModule, MuZeroNet, scale_gradient
from .replay import Outcome, PolicyHead
from .storage import atomic_path


def create_model(config):
    return MuZeroNet(config['CANVAS_SIZE'], INPUT_PLANES, config['NUM_BLOCKS'], config['NUM_CHANNELS'],
                     config['VALUE_HEAD'], config['AUXILIARY_POLICY_HEADS'])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def export_model(model, path):
    scripted = torch.jit.script(InferenceModule(copy.deepcopy(model).cpu().eval()))
    with atomic_path(path) as temporary:
        torch.jit.save(scripted, str(temporary))


def create_optimizer(model, c):
    return torch.optim.AdamW(model.parameters(), lr=c['LR'], weight_decay=c['WEIGHT_DECAY'],
                             betas=(c['ADAM_BETA1'], c['ADAM_BETA2']), eps=c['ADAM_EPS'])


def save_checkpoint(path, model, optimizer, config, iteration, metrics):
    with atomic_path(path) as temporary:
        torch.save({'protocol_version': VERSION, 'identity': model_identity(config), 'iteration': iteration,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'metrics': metrics}, temporary)


def load_checkpoint(path, model, optimizer, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['protocol_version'] != VERSION or checkpoint['identity'] != model_identity(config):
        raise ValueError('Checkpoint model/data protocol mismatch; use a new DATA_DIR')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def train_iteration(model, optimizer, replay, config, iteration, device):
    seed = config['SEED'] + iteration + 1
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model.train()
    scales = torch.ones(model.prediction.policy_count, device=device)
    if config['AUXILIARY_POLICY_HEADS']:
        scales[PolicyHead.SOFT] = config['SOFT_POLICY_LOSS_SCALE']
        scales[PolicyHead.OPPONENT] = config['OPPONENT_POLICY_LOSS_SCALE']
        scales[PolicyHead.SOFT_OPPONENT] = config['SOFT_POLICY_LOSS_SCALE'] * config['OPPONENT_POLICY_LOSS_SCALE']
    totals = np.zeros(3, dtype=np.float64)
    groups = {}
    for _ in range(config['TRAIN_STEPS']):
        samples = replay.sample(rng)
        observations, actions, policies, values, masks = [torch.from_numpy(array).to(device) for array in samples]
        optimizer.zero_grad(set_to_none=True)
        hidden = model.representation(observations)
        policy_loss = torch.zeros((), device=device)
        value_loss = torch.zeros((), device=device)
        sample_losses = torch.zeros((config['BATCH_SIZE'], 2), device=device)
        for step in range(config['UNROLL_STEPS'] + 1):
            policy_logits, value_logits = model.prediction(hidden)
            policy_rows = (-(policies[:, step] * F.log_softmax(policy_logits, -1)).sum(-1) * masks[:, step] * scales).sum(-1)
            if model.prediction.wdl:
                value_rows = -(values[:, step] * F.log_softmax(value_logits, -1)).sum(-1)
            else:
                targets = values[:, step, Outcome.WIN] - values[:, step, Outcome.LOSS]
                value_rows = F.mse_loss(model.prediction.utility(value_logits), targets, reduction='none')
            sample_losses += torch.stack([policy_rows.detach(), value_rows.detach()], dim=1)
            gradient_scale = 1.0 if step == 0 else 1.0 / config['UNROLL_STEPS']
            policy_loss = policy_loss + scale_gradient(policy_rows.sum(), gradient_scale)
            value_loss = value_loss + scale_gradient(value_rows.sum(), gradient_scale)
            if step < config['UNROLL_STEPS']:
                if step > 0:
                    hidden = scale_gradient(hidden, config['HIDDEN_GRADIENT_SCALE'])
                hidden = model.dynamics(hidden, actions[:, step])
        policy_loss = policy_loss / config['BATCH_SIZE']
        value_loss = value_loss / config['BATCH_SIZE']
        loss = policy_loss + config['VALUE_LOSS_SCALE'] * value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError('Nonfinite training gradient')
        optimizer.step()
        totals += [loss.item(), policy_loss.item(), value_loss.item()]
        for obs, row_losses in zip(samples[0], sample_losses.cpu().numpy()):
            size = int(round(np.sqrt(obs[Plane.ON_BOARD].sum())))
            rule = 'renju' if obs[Plane.RENJU].any() else 'standard' if obs[Plane.STANDARD].any() else 'freestyle'
            group = groups.setdefault(f'{rule}/{size}', dict(samples=0, policy_loss=0.0, value_loss=0.0))
            group['samples'] += 1
            group['policy_loss'] += float(row_losses[0])
            group['value_loss'] += float(row_losses[1])
    model.eval()
    totals /= config['TRAIN_STEPS']
    for group in groups.values():
        group['policy_loss'] /= group['samples']
        group['value_loss'] /= group['samples']
    return dict(zip(('loss', 'policy_loss', 'value_loss'), totals.tolist())) | {'steps': config['TRAIN_STEPS'], 'train_groups': groups}
