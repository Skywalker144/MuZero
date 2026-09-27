import copy
import random
import time
import math

import numpy as np
import torch
import torch.nn.functional as F

from .protocol import VERSION, INPUT_PLANES, Plane
from .config import model_identity
from .model_config import ModelConfig
from .network import InferenceModule, MuZeroNet, scale_gradient
from .replay import Outcome, PolicyHead
from .storage import atomic_path
from .prefetch import BatchStream


class WeightAverage:
    def __init__(self, model, config):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.halflife = config['EMA_HALFLIFE_SAMPLES']
        self.samples = 0

    @torch.no_grad()
    def update(self, model, samples):
        alpha = 1.0 if self.samples == 0 else -math.expm1(-math.log(2) * samples / self.halflife)
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.lerp_(current.detach(), alpha)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)
        self.samples += samples

    def state_dict(self):
        return {'model': self.model.state_dict(), 'samples': self.samples}

    def load_state_dict(self, state):
        if type(state['samples']) is not int or state['samples'] < 0:
            raise ValueError('Invalid EMA sample count')
        self.model.load_state_dict(state['model'])
        self.samples = state['samples']


def create_model(config):
    return MuZeroNet(config['CANVAS_SIZE'], INPUT_PLANES, ModelConfig.from_mapping(config))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def export_model(model, path):
    scripted = torch.jit.script(InferenceModule(copy.deepcopy(model).cpu().eval()))
    with atomic_path(path) as temporary:
        torch.jit.save(scripted, str(temporary))
        restored = torch.jit.load(str(temporary))
        canvas, version = restored.metadata()
        if version != VERSION:
            raise ValueError('Export protocol mismatch')
        with torch.inference_mode():
            observation = torch.zeros(2, INPUT_PLANES, canvas, canvas)
            observation[:, Plane.ON_BOARD] = 1
            observation[:, Plane.BLACK_TO_MOVE] = 1
            initial = restored.initial(observation)
            recurrent = restored.recurrent(initial[0], torch.zeros(2, dtype=torch.int64))
            for actual, expected in ((initial, scripted.initial(observation)),
                                     (recurrent, scripted.recurrent(initial[0], torch.zeros(2, dtype=torch.int64)))):
                for value, reference in zip(actual, expected):
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite exported inference')
                    torch.testing.assert_close(value, reference)


def create_optimizer(model, c):
    return torch.optim.AdamW(model.parameters(), lr=c['LR'], weight_decay=c['WEIGHT_DECAY'],
                             betas=(c['ADAM_BETA1'], c['ADAM_BETA2']), eps=c['ADAM_EPS'])


def training_steps(checkpoint):
    steps = checkpoint.get('training_steps')
    if type(steps) is not int or steps < 0:
        raise ValueError('Missing or invalid training_steps in model checkpoint; use a new initialization or DATA_DIR')
    return steps


def save_checkpoint(path, model, optimizer, config, iteration, metrics, steps, average):
    with atomic_path(path) as temporary:
        torch.save({'protocol_version': VERSION, 'identity': model_identity(config), 'iteration': iteration,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'metrics': metrics,
                    'training_steps': steps, 'average': average.state_dict()}, temporary)


def load_checkpoint(path, model, optimizer, config, average):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['protocol_version'] != VERSION or checkpoint['identity'] != model_identity(config):
        raise ValueError('Checkpoint model/data protocol mismatch; use a new DATA_DIR')
    training_steps(checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    average.load_state_dict(checkpoint['average'])
    return checkpoint


def train_iteration(model, optimizer, replay, config, iteration, device, average):
    seed = config['SEED'] + iteration + 1
    seed_all(seed)
    model.train()
    scales = torch.ones(model.prediction.policy_count, device=device)
    if config['AUXILIARY_POLICY_HEADS']:
        scales[PolicyHead.SOFT] = config['SOFT_POLICY_LOSS_SCALE']
        scales[PolicyHead.OPPONENT] = config['OPPONENT_POLICY_LOSS_SCALE']
        scales[PolicyHead.SOFT_OPPONENT] = config['SOFT_POLICY_LOSS_SCALE'] * config['OPPONENT_POLICY_LOSS_SCALE']
    totals = np.zeros(3, dtype=np.float64)
    step_totals = torch.zeros(config['UNROLL_STEPS'] + 1, device=device)
    gradient_totals = {name: torch.zeros((), device=device)
                       for name in ('representation', 'dynamics', 'prediction')}
    groups = {}
    data_wait_seconds = 0.0
    training_started = time.monotonic()
    with BatchStream(replay, seed, config['TRAIN_STEPS'], config['BATCH_PREFETCH'], device) as batches:
        for _ in range(config['TRAIN_STEPS']):
            waiting = time.monotonic()
            samples = next(batches)
            data_wait_seconds += time.monotonic() - waiting
            observations, actions, policies, values, masks = [tensor.to(device, non_blocking=True) for tensor in samples]
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
                step_totals[step] += policy_rows.detach().mean() + config['VALUE_LOSS_SCALE'] * value_rows.detach().mean()
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
            finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                                if parameter.grad is not None]
            if finite_gradients and not torch.stack(finite_gradients).all():
                raise FloatingPointError('Nonfinite training gradient')
            for name, total in gradient_totals.items():
                norms = [parameter.grad.detach().norm() for parameter in getattr(model, name).parameters()
                         if parameter.grad is not None]
                if norms:
                    total += torch.stack(norms).norm()
            optimizer.step()
            average.update(model, config['BATCH_SIZE'])
            totals += [loss.item(), policy_loss.item(), value_loss.item()]
            for obs, row_losses in zip(samples[0].numpy(), sample_losses.cpu().numpy()):
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
    return dict(zip(('loss', 'policy_loss', 'value_loss'), totals.tolist())) | {
        'steps': config['TRAIN_STEPS'], 'train_groups': groups,
        'batch_prepare_seconds': batches.prepare_seconds, 'data_wait_seconds': data_wait_seconds,
        'train_loop_seconds': time.monotonic() - training_started,
        'ema_samples': average.samples,
        'step_losses': (step_totals / config['TRAIN_STEPS']).tolist(),
        'grad_norms': {name: (total / config['TRAIN_STEPS']).item() for name, total in gradient_totals.items()},
    }
