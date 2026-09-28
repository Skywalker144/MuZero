import unittest
import numpy as np

from muzero.elo import fit_ratings


class EloTests(unittest.TestCase):
    def test_joint_fit_recovers_rating_and_first_player_advantage(self):
        rng = np.random.default_rng(81)
        games = []
        strengths = {'a': 0, 'b': 150, 'c': -100}
        for a, b in [('a', 'b'), ('a', 'c'), ('b', 'c')]:
            for opening in range(1500):
                for black_a in (True, False):
                    black, white = (a, b) if black_a else (b, a)
                    p = 1 / (1 + 10 ** (-(strengths[black] - strengths[white] + 60) / 400))
                    games.append({'pair': a+b, 'opening_id': opening, 'black': black,
                                  'white': white, 'score': float(rng.random() < p)})
        result = fit_ratings(games, 'a', samples=30, seed=1)
        ratings = {r['player']: r for r in result['ratings']}
        self.assertEqual(ratings['a']['elo'], 0)
        self.assertEqual(ratings['a']['lower'], 0)
        for player, strength in strengths.items():
            self.assertAlmostEqual(ratings[player]['elo'], strength, delta=25)
            self.assertLessEqual(ratings[player]['lower'], ratings[player]['upper'])
        self.assertAlmostEqual(result['first_player_advantage_elo'], 60, delta=20)

    def test_disconnected_graph_is_rejected(self):
        games = [{'pair': a+b, 'opening_id': 0, 'black': a, 'white': b, 'score': 1}
                 for a, b in [('a', 'b'), ('c', 'd')]]
        with self.assertRaisesRegex(ValueError, 'connected'):
            fit_ratings(games, 'a', samples=10)

    def test_sweep_and_draws_stay_finite(self):
        games = []
        for opening in range(20):
            games.extend([
                {'pair': 'ab', 'opening_id': opening, 'black': 'a', 'white': 'b', 'score': 1},
                {'pair': 'ab', 'opening_id': opening, 'black': 'b', 'white': 'a', 'score': 0},
                {'pair': 'bc', 'opening_id': opening, 'black': 'b', 'white': 'c', 'score': 0.5},
                {'pair': 'bc', 'opening_id': opening, 'black': 'c', 'white': 'b', 'score': 0.5},
            ])
        result = fit_ratings(games, 'b', samples=10)
        self.assertTrue(all(np.isfinite(r['elo']) for r in result['ratings']))
        self.assertGreater(next(r['elo'] for r in result['ratings'] if r['player'] == 'a'), 0)
        self.assertTrue(result['separated_pairs'])


if __name__ == '__main__':
    unittest.main()
