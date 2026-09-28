import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .storage import atomic_path, write_json


ELO_SCALE = 400 / np.log(10)
PRIOR_SIGMA_ELO = 1000.0


def fit_ratings(games: list[dict], anchor: str, samples: int = 300, seed: int = 1) -> dict:
    if not games or samples < 2:
        raise ValueError('Need completed games and at least two bootstrap samples')
    players = sorted({g[side] for g in games for side in ('black', 'white')})
    if anchor not in players:
        raise ValueError('Anchor has no completed games')
    adjacent = {p: set() for p in players}
    for game in games:
        if game['black'] == game['white'] or game['score'] not in (0, 0.5, 1):
            raise ValueError('Invalid game score or opponent')
        adjacent[game['black']].add(game['white'])
        adjacent[game['white']].add(game['black'])
    reached, pending = set(), [anchor]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(adjacent[node] - reached)
    if reached != set(players):
        raise ValueError('Match graph must be connected before a joint Elo fit')
    free = [p for p in players if p != anchor]
    indices = {p: i for i, p in enumerate(free)}
    ordered = sorted({(g['black'], g['white']) for g in games})
    order_index = {pair: i for i, pair in enumerate(ordered)}
    design = np.zeros((len(ordered), len(players)))
    for row, (black, white) in enumerate(ordered):
        if black != anchor:
            design[row, indices[black]] = 1
        if white != anchor:
            design[row, indices[white]] = -1
        design[row, -1] = 1
    counts = np.zeros(len(ordered))
    wins = np.zeros(len(ordered))
    clusters = defaultdict(lambda: defaultdict(list))
    records = {p: {'games': 0, 'wins': 0, 'draws': 0, 'losses': 0} for p in players}
    pair_scores = defaultdict(list)
    for game in games:
        row = order_index[(game['black'], game['white'])]
        score = game['score']
        counts[row] += 1
        wins[row] += score
        clusters[game['pair']][game['opening_id']].append((row, score))
        a, b = sorted((game['black'], game['white']))
        pair_scores[(a, b)].append(score if game['black'] == a else 1-score)
        for player, result in ((game['black'], score), (game['white'], 1-score)):
            records[player]['games'] += 1
            records[player]['wins' if result == 1 else 'losses' if result == 0 else 'draws'] += 1
    penalty = np.eye(len(players))
    penalty[:-1, :-1] -= np.ones((len(free), len(free))) / len(players)
    penalty *= (ELO_SCALE / PRIOR_SIGMA_ELO) ** 2

    def solve(n, w, initial):
        def objective(theta):
            logits = design @ theta
            prior = penalty @ theta
            return (np.sum(n * np.logaddexp(0, logits) - w * logits) + 0.5 * theta @ prior,
                    design.T @ (n * expit(logits) - w) + prior)
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 2000})
        if not result.success:
            raise ValueError(f'Elo optimization failed: {result.message}')
        return result.x

    fitted = solve(counts, wins, np.zeros(len(players)))
    blocks = []
    incomplete = 0
    for openings in clusters.values():
        block_n = np.zeros((len(openings), len(ordered)))
        block_w = np.zeros_like(block_n)
        for i, results in enumerate(openings.values()):
            incomplete += (len(results) != 2 or
                           ordered[results[0][0]] != tuple(reversed(ordered[results[-1][0]])))
            for row, score in results:
                block_n[i, row] += 1
                block_w[i, row] += score
        blocks.append((block_n, block_w))
    if incomplete:
        raise ValueError('Complete both colors of each opening before fitting')
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        n, w = np.zeros_like(counts), np.zeros_like(wins)
        for block_n, block_w in blocks:
            multiplicity = rng.multinomial(len(block_n), np.full(len(block_n), 1 / len(block_n)))
            n += multiplicity @ block_n
            w += multiplicity @ block_w
        bootstrap.append(solve(n, w, fitted))
    bootstrap = np.asarray(bootstrap) * ELO_SCALE
    lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    rows = []
    for player in players:
        i = indices.get(player)
        rows.append({'player': player, 'elo': float(fitted[i] * ELO_SCALE) if i is not None else 0.0,
                     'lower': float(lower[i]) if i is not None else 0.0,
                     'upper': float(upper[i]) if i is not None else 0.0, **records[player]})
    separated = [{'a': a, 'b': b, 'games': len(scores), 'score_a': sum(scores)}
                 for (a, b), scores in pair_scores.items() if sum(scores) in (0, len(scores))]
    return {'anchor': anchor, 'games': len(games), 'ratings': rows,
            'first_player_advantage_elo': float(fitted[-1] * ELO_SCALE),
            'first_player_advantage_interval': [float(lower[-1]), float(upper[-1])],
            'interval_method': '95% stratified paired-opening bootstrap', 'bootstrap_samples': samples,
            'prior_sigma_elo': PRIOR_SIGMA_ELO, 'separated_pairs': separated}


def write_ratings(output: Path, samples: int = 300) -> dict:
    from .arena import load_results
    manifest, games = load_results(output)
    expected = len(manifest['pairs']) * manifest['games_per_pair']
    if len(games) != expected:
        raise ValueError(f'Incomplete schedule: {len(games)}/{expected} games; resume the arena first')
    result = fit_ratings(games, manifest['anchor'], samples, manifest['config']['SEED'])
    metadata = {p['id']: p for p in manifest['players']}
    for row in result['ratings']:
        row.update({k: v for k, v in metadata[row['player']].items() if k != 'id'})
    result['config'] = manifest['config']
    write_json(output / 'elo.json', result)
    with atomic_path(output / 'elo.csv') as temporary:
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['ratings'][0]))
            writer.writeheader()
            writer.writerows(result['ratings'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(11, 6.5))
    arms = defaultdict(list)
    for row in result['ratings']:
        arms[row['arm']].append(row)
    colors = ('#2764A5', '#C27B19', '#A34473', '#667C3E', '#696969')
    markers = ('o', 's', '^', 'D', 'v')
    for index, (arm, rows) in enumerate(sorted(arms.items())):
        rows.sort(key=lambda r: r['seconds'])
        hours = [r['seconds'] / 3600 for r in rows]
        line, = axis.plot(hours, [r['elo'] for r in rows], marker=markers[index % len(markers)],
                          color=colors[index % len(colors)], markersize=4, label=arm)
        axis.fill_between(hours, [r['lower'] for r in rows], [r['upper'] for r in rows],
                          color=line.get_color(), alpha=0.16)
    config = manifest['config']
    axis.set(xlabel='Cumulative committed training wall time (hours)',
             ylabel=f"Relative Elo ({manifest['anchor']} = 0)",
             title=f"{config['BOARD_SIZE']}×{config['BOARD_SIZE']} {config['RULE'].capitalize()} · "
                   f"{config['FULL_SEARCH_VISITS']} visits · {len(games):,} games")
    axis.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    note = 'Shading: 95% paired-opening bootstrap; fixed evaluation search for every model.'
    if result['separated_pairs']:
        note += '\nSwept pairings present: rating magnitudes and intervals can be prior-sensitive.'
    figure.text(0.5, 0.015, note, ha='center', fontsize=8)
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    for suffix in ('png', 'svg'):
        with atomic_path(output / f'elo.{suffix}') as temporary:
            figure.savefig(temporary, format=suffix, dpi=160)
    plt.close(figure)
    print(f"Elo: {output / 'elo.png'}", flush=True)
    return result
