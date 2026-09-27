import tempfile
import unittest
from pathlib import Path

from muzero.config import ROOT, load_config, load_eval_config, render_config, render_native_config


class EvalConfigTests(unittest.TestCase):
    def test_defaults_are_independent_of_selfplay(self):
        c = load_eval_config(ROOT / 'configs/exp_muzero', environ={
            'FULL_SEARCH_VISITS': '7', 'DEVICE': 'cpu', 'USE_FPU': 'false',
        })
        self.assertEqual(c['FULL_SEARCH_VISITS'], 400)
        self.assertEqual(c['DEVICE'], 'cuda:0')
        self.assertTrue(c['USE_FPU'])
        self.assertTrue(c['USE_LCB_FOR_SELECTION'])
        self.assertEqual(c['ROOT_FPU_REDUCTION_MAX'], 0.1)
        self.assertEqual(c['NUM_SEARCH_THREADS'], 8)
        self.assertEqual(c['VIRTUAL_LOSS'], 1.0)
        self.assertEqual(c['CHOSEN_MOVE_TEMPERATURE'], 0)
        self.assertNotIn('CHEAP_SEARCH_PROB', c)
        self.assertNotIn('LR', c)
        self.assertIn('VIRTUAL_LOSS=1.0', render_native_config(c))

    def test_inheritance_local_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text('extends = baseline\n')
            (path / 'eval.cfg').write_text('[search]\nfull_search_visits = 80\n')
            (path / 'eval.cfg.local').write_text('[search]\nfull_search_visits = 60\n')
            self.assertEqual(load_eval_config(path, environ={})['FULL_SEARCH_VISITS'], 60)
            c = load_eval_config(path, environ={'EVAL_FULL_SEARCH_VISITS': '20', 'EVAL_DEVICE': 'cpu'})
            self.assertEqual(c['FULL_SEARCH_VISITS'], 20)
            self.assertEqual(c['DEVICE'], 'cpu')
            self.assertEqual(load_config(path, environ={})['FULL_SEARCH_VISITS'], 400)
            (path / 'eval.cfg.local').unlink()
            (path / 'eval.cfg').write_text(render_config(c, 'eval.cfg'))
            self.assertEqual(load_eval_config(path, environ={}), c)

    def test_invalid_eval_parameters(self):
        for key, value in [('NUM_SEARCH_THREADS', '0'), ('VIRTUAL_LOSS', '-1'),
                           ('VIRTUAL_LOSS', 'nan'), ('FULL_SEARCH_VISITS', '0'),
                           ('ROOT_POLICY_TEMPERATURE', '0'), ('CHOSEN_MOVE_TEMPERATURE', '-1'),
                           ('MIN_VISIT_PROP_FOR_LCB', '2'), ('DEVICE', 'typo')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_eval_config('configs/baseline', environ={f'EVAL_{key}': value})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text('extends = baseline\n')
            for contents in ('[noise]\ndirichlet_noise_weight = 0.2\n',
                             '[search]\ncheap_search_prob = 0.5\n',
                             '[parallel]\nnum_search_threads = 4\n',
                             'extends = baseline\n'):
                (path / 'eval.cfg').write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_eval_config(path, environ={})


if __name__ == '__main__':
    unittest.main()
