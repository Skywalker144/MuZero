from .mcts import MCTS, Search
from .selfplay import GameHistory


class _GameSession:
    def __init__(self, game, config):
        self.history = GameHistory(game, config)
        self.search: Search | None = None

    def next_leaf(self):
        if self.search is not None:
            node = self.search.next_leaf()
            if node is not None:
                return node
            self.history.play(self.search.result(), self.search.budget)
            self.search.close()
            self.search = None
        if self.history.winner is not None:
            return None
        history = self.history
        self.search = Search(
            history.game, history.state, history.to_play, history.config,
            history.config.sample_budget(), len(history.steps),
        )
        return self.search.next_leaf()


class ParallelSelfPlayer:
    def __init__(self, game, args, model, device):
        self.mcts = MCTS(game, args, model, device)
        self.num_parallel_games = max(1, int(args.get("num_parallel_games", 32)))

    def run(self, total_games):
        active = []
        started = 0
        try:
            while started < total_games or active:
                while started < total_games and len(active) < self.num_parallel_games:
                    active.append(_GameSession(self.mcts.game, self.mcts.config))
                    started += 1
                requests = []
                finished = []
                remaining = []
                for session in active:
                    node = session.next_leaf()
                    if node is None:
                        finished.append(session.history.result())
                    else:
                        remaining.append(session)
                        requests.append((session, node))
                active = remaining
                if requests:
                    results = self.mcts.infer_nodes([node for _, node in requests])
                    for (session, node), (logits, value) in zip(requests, results):
                        session.search.complete(node, logits, value)
                yield from finished
        finally:
            for session in active:
                if session.search is not None:
                    session.search.close()
