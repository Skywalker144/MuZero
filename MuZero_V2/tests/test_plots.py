import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import image
from matplotlib.figure import Figure

from muzero.plots import training_figure, render_training
from muzero.storage import write_json


class PlotTests(unittest.TestCase):
    def test_sample_ticks_and_policy_head_losses(self):
        history = [dict(iteration=0, games=1, rows=700000, total_rows=700000,
                        black_wins=1, white_wins=0, draws=0, steps=1, loss=5.,
                        policy_loss=4., value_loss=1.,
                        policy_head_losses=dict(main=1., soft=2., opponent=.3, soft_opponent=.7))]
        figure = training_figure(history)
        for axis in figure.axes[:2]:
            formatter = axis.xaxis.get_major_formatter()
            for value, label in [(0, '0'), (700000, '7e5'), (640000, '6.4e5'), (1000, '1e3')]:
                self.assertEqual(formatter(value), label)
        lines = {line.get_label(): line for line in figure.axes[3].lines}
        for label, value in [('Policy main', 1.), ('Policy soft', 2.),
                             ('Policy opponent', .3), ('Policy soft opponent', .7)]:
            np.testing.assert_allclose(lines[label].get_ydata(), [value])

    def test_empty_and_bootstrap_history(self):
        for history in ([], [dict(iteration=0, games=2, rows=30, total_rows=30,
                                  black_wins=1, white_wins=0, draws=1, steps=0,
                                  loss=None, policy_loss=None, value_loss=None)]):
            with self.subTest(history=history):
                figure = training_figure(history)
                self.assertEqual(len(figure.axes), 6)
                self.assertFalse(figure.axes[2].lines)
                self.assertFalse(figure.axes[4].lines)
                self.assertFalse(figure.axes[5].lines)
                if history:
                    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0.5])
                    np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [15])

    def test_metric_coordinates_gaps_and_zero_gradients(self):
        history = [
            dict(iteration=0, games=2, rows=30, total_rows=30, black_wins=1,
                 white_wins=0, draws=1, steps=0, loss=None, policy_loss=None, value_loss=None),
            dict(iteration=1, games=2, rows=40, total_rows=70, black_wins=0,
                 white_wins=2, draws=0, steps=2, loss=3., policy_loss=2., value_loss=1.,
                 step_losses=[2., 1.], grad_norms=dict(representation=2., dynamics=0., prediction=3.)),
            dict(iteration=2, games=0, rows=0, total_rows=70, black_wins=0,
                 white_wins=0, draws=0, steps=2, loss=2., policy_loss=1., value_loss=1.,
                 step_losses=[1.5, .5], grad_norms=dict(representation=1., dynamics=0., prediction=2.)),
        ]
        figure = training_figure(history)
        outcomes, lengths, total, components, steps, gradients = figure.axes
        np.testing.assert_allclose(outcomes.lines[0].get_xdata(), [30, 70])
        np.testing.assert_allclose(outcomes.lines[0].get_ydata(), [.5, 0])
        np.testing.assert_allclose(lengths.lines[0].get_ydata(), [15, 20])
        np.testing.assert_allclose(total.lines[0].get_xdata(), [2, 3])
        np.testing.assert_allclose(total.lines[0].get_ydata(), [3, 2])
        np.testing.assert_allclose(components.lines[0].get_ydata(), [2, 1])
        np.testing.assert_allclose(steps.lines[0].get_ydata(), [1.75, .75])
        np.testing.assert_allclose(steps.lines[1].get_ydata(), [1.5, .5])
        np.testing.assert_allclose(gradients.lines[1].get_ydata(), [0, 0])
        self.assertEqual(gradients.get_yscale(), 'linear')
        self.assertEqual(total.get_yscale(), 'log')

    def test_render_from_logs_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            metrics = dict(iteration=0, games=2, rows=30, total_rows=30,
                           black_wins=1, white_wins=0, draws=1, steps=0,
                           loss=None, policy_loss=None, value_loss=None)
            path = data / 'logs/iters/00000000.json'
            write_json(path, metrics)
            output = render_training(data)
            self.assertEqual(output, data / 'training.png')
            pixels = image.imread(output)
            self.assertGreater(pixels.shape[0], 1000)
            self.assertGreater(pixels.std(), .05)
            original = output.read_bytes()
            with patch.object(Figure, 'savefig', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    render_training(data)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(data.glob('*.tmp')), [])
            self.assertEqual(json.loads(path.read_text()), metrics)


if __name__ == '__main__':
    unittest.main()
