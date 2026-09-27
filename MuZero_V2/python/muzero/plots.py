import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np

from .config import ROOT
from .storage import atomic_path


BLUE, RED, ORANGE, GREEN, GREY = '#61afef', '#e06c75', '#e8924b', '#98c379', '#9aa2b1'
THEME = {
    'figure.facecolor': '#1e2127', 'axes.facecolor': '#282c34',
    'savefig.facecolor': '#1e2127', 'text.color': '#abb2bf',
    'axes.labelcolor': '#abb2bf', 'axes.titlecolor': '#c8ccd4',
    'axes.edgecolor': '#454c5a', 'xtick.color': '#8a93a3', 'ytick.color': '#8a93a3',
    'grid.color': '#3b4250', 'legend.labelcolor': '#abb2bf', 'legend.frameon': False,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'grid.linewidth': .7,
}


def _plot_series(axis, series: Sequence[tuple[str, str, Sequence, Sequence]], *, logarithmic: bool = False):
    positive = True
    for label, color, x, y in series:
        values = np.asarray(y, dtype=np.float64)
        if not np.isfinite(values).any():
            continue
        positive &= bool((values[np.isfinite(values)] > 0).all())
        axis.plot(x, values, color=color, linewidth=1.7, label=label)
    if axis.lines:
        if logarithmic and positive:
            axis.set_yscale('log')
        axis.legend(loc='best', fontsize=9)
    else:
        axis.text(.5, .5, 'No measurements yet', ha='center', va='center',
                  color='#8a93a3', transform=axis.transAxes)


def training_figure(history: Sequence[Mapping[str, Any]]) -> Figure:
    with mpl.rc_context(THEME):
        figure = Figure(figsize=(15, 13.5), layout='constrained')
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 2).flat
        figure.suptitle('MuZero training progress', fontsize=17)
        games = sum(row['games'] for row in history)
        samples = history[-1]['total_rows'] if history else 0
        figure.supxlabel(f'{len(history)} iterations · {games:,} self-play games · '
                         f'{samples:,} self-play samples', fontsize=10, color='#8a93a3')
        for axis, title, xlabel, ylabel in zip(axes, (
            'Self-play outcomes', 'Self-play game length', 'Total loss', 'Loss components',
            'Loss by unroll step', 'Gradient norms by module',
        ), (
            'Cumulative self-play samples (iteration end)', 'Cumulative self-play samples (iteration end)',
            'Iteration (1-based)', 'Iteration (1-based)', 'Unroll step k', 'Iteration (1-based)',
        ), ('Share of games', 'Moves per game', 'Loss', 'Loss', 'Loss per step', 'L2 grad norm')):
            axis.set_title(title, pad=12)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(axis='y')
            axis.set_axisbelow(True)
            axis.spines[['top', 'right']].set_visible(False)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        selfplay = [row for row in history if row['games'] > 0]
        x = [row['total_rows'] for row in selfplay]
        _plot_series(axes[0], [
            (label, color, x, [row[key] / row['games'] for row in selfplay])
            for key, label, color in (('black_wins', 'Black win', BLUE),
                                      ('white_wins', 'White win', RED), ('draws', 'Draw', GREY))
        ])
        axes[0].set_ylim(0, 1.025)
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        _plot_series(axes[1], [('Mean', ORANGE, x, [row['rows'] / row['games'] for row in selfplay])])
        for axis in (axes[0], axes[1]):
            axis.set_xlim(0, max(1, samples * 1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000:g}k' if abs(value) >= 1000 else f'{value:g}'))

        trained = [row for row in history if row['steps'] > 0]
        x = [row['iteration'] + 1 for row in trained]
        _plot_series(axes[2], [('Total', ORANGE, x, [row['loss'] for row in trained])], logarithmic=True)
        _plot_series(axes[3], [(label, color, x, [row[key] for row in trained])
                               for key, label, color in (('policy_loss', 'Policy', BLUE), ('value_loss', 'Value', GREEN))],
                     logarithmic=True)
        step_losses = [row['step_losses'] for row in trained if row.get('step_losses')]
        series = []
        if step_losses:
            matrix = np.asarray(step_losses, dtype=np.float64)
            steps = np.arange(matrix.shape[1])
            series = [('Mean over iterations', ORANGE, steps, matrix.mean(axis=0)),
                      ('Latest iteration', BLUE, steps, matrix[-1])]
        _plot_series(axes[4], series, logarithmic=True)
        _plot_series(axes[5], [
            (label, color, x, [row.get('grad_norms', {}).get(key, np.nan) for row in trained])
            for key, label, color in (('representation', 'Representation h', BLUE),
                                     ('dynamics', 'Dynamics g', GREEN), ('prediction', 'Prediction f', ORANGE))
        ], logarithmic=True)
    return figure


def render_training(data: Path) -> Path:
    data = Path(data)
    history = [json.loads(path.read_text()) for path in sorted((data / 'logs/iters').glob('*.json'))]
    figure = training_figure(history)
    output = data / 'training.png'
    try:
        with atomic_path(output) as temporary:
            figure.savefig(temporary, format='png', dpi=160, facecolor=THEME['figure.facecolor'])
    finally:
        figure.clear()
    return output


def main():
    parser = argparse.ArgumentParser(description='Rebuild training.png from iteration logs')
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()
    data = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    if not (data / 'logs/iters').is_dir():
        parser.error(f'No iteration logs found in {data / "logs/iters"}')
    print(render_training(data))


if __name__ == '__main__':
    main()
