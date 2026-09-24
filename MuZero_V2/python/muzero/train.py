import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import model_identity
from .network import InferenceModule, MuZeroNet, scale_gradient
from .replay import Outcome, PolicyHead
from .storage import atomic_path


def create_model(config):
    return MuZeroNet(config['BOARD_SIZE'], 3, config['NUM_BLOCKS'], config['NUM_CHANNELS'],
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
        torch.save({'protocol_version': 2, 'identity': model_identity(config), 'iteration': iteration,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'metrics': metrics}, temporary)


def load_checkpoint(path, model, optimizer, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['protocol_version'] != 2 or checkpoint['identity'] != model_identity(config):
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
    for _ in range(config['TRAIN_STEPS']):
        observations, actions, policies, values, masks = [torch.from_numpy(array).to(device) for array in replay.sample(rng)]
        optimizer.zero_grad(set_to_none=True)
        hidden = model.representation(observations)
        policy_loss = torch.zeros((), device=device)
        value_loss = torch.zeros((), device=device)
        for step in range(config['UNROLL_STEPS'] + 1):
            policy_logits, value_logits = model.prediction(hidden)
            policy_term = (-(policies[:, step] * F.log_softmax(policy_logits, -1)).sum(-1) * masks[:, step] * scales).sum()
            if model.prediction.wdl:
                value_term = -(values[:, step] * F.log_softmax(value_logits, -1)).sum()
            else:
                targets = values[:, step, Outcome.WIN] - values[:, step, Outcome.LOSS]
                value_term = F.mse_loss(model.prediction.utility(value_logits), targets, reduction='sum')
            gradient_scale = 1.0 if step == 0 else 1.0 / config['UNROLL_STEPS']
            policy_loss = policy_loss + scale_gradient(policy_term, gradient_scale)
            value_loss = value_loss + scale_gradient(value_term, gradient_scale)
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
    model.eval()
    totals /= config['TRAIN_STEPS']
    return dict(zip(('loss', 'policy_loss', 'value_loss'), totals.tolist())) | {'steps': config['TRAIN_STEPS']}
