import unittest
import tempfile
from pathlib import Path

from muzero.config import CONFIG_FILES, ROOT, load_config, render_config, render_native_config


class MultiGameConfigTests(unittest.TestCase):
    def test_weighted_games(self):
        c = load_config(ROOT / 'configs/baseline', environ={
            'BOARD_SIZES': '5,7,9', 'BOARD_SIZE_WEIGHTS': '1,2,3',
            'RULES': 'freestyle,standard,renju', 'RULE_WEIGHTS': '1,1,2',
        })
        self.assertEqual(c['BOARD_SIZES'], [5, 7, 9])
        self.assertEqual(c['CANVAS_SIZE'], 9)
        self.assertIn('BOARD_SIZES=5,7,9\n', render_native_config(c))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for filename in CONFIG_FILES:
                (directory / filename).write_text(render_config(c, filename))
            self.assertEqual(load_config(directory, environ={}), c)

    def test_invalid_distributions(self):
        for values in (
            {'BOARD_SIZES': '4'}, {'BOARD_SIZES': '26'},
            {'BOARD_SIZES': '5,5', 'BOARD_SIZE_WEIGHTS': '1,1'},
            {'BOARD_SIZES': '5,7'}, {'BOARD_SIZE_WEIGHTS': '0'},
            {'RULES': 'unknown'}, {'RULE_WEIGHTS': '-1'}, {'RULE_WEIGHTS': 'nan'},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_config(ROOT / 'configs/baseline', environ=values)
