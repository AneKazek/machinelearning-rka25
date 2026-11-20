"""Local training environment for the Pac-Man Q-learning experiment."""

from collections import deque
from pathlib import Path
from random import Random
import json


# Action order also determines how the browser resolves equal distances/values.
ACTIONS = ("up", "right", "down", "left")
OFFSETS = ((0, -1), (1, 0), (0, 1), (-1, 0))
MAZE = (
    "###########",
    "#.........#",
    "#.#.#.###.#",
    "#...#.....#",
    "###...#.#.#",
    "#...#.....#",
    "#.#.#.##..#",
    "#.#...#...#",
    "#...#...#.#",
    "#.........#",
    "###########",
)
TILES = tuple((x, y) for y, row in enumerate(MAZE)
              for x, tile in enumerate(row) if tile == ".")
GRAPH = {
    tile: tuple((action, (tile[0] + dx, tile[1] + dy))
                for action, (dx, dy) in enumerate(OFFSETS)
                if (tile[0] + dx, tile[1] + dy) in TILES)
    for tile in TILES
}
VALID = {tile: sum(1 << action for action, _ in edges)
         for tile, edges in GRAPH.items()}
STEP_LIMIT = 150
POWER_TILES = {(1, 1), (9, 1), (1, 9), (9, 9)}
FRUIT_TILE = (5, 5)
POWER_TURNS = 18
FRUIT_TURNS = 30
POWER, FRUIT, EDIBLE_GHOST = 1, 2, 3
CAUGHT_PENALTY = 25


def routes_from(start):
    """BFS insertion order resolves shortest-path ties in action order."""
    routes = {start: (0, -1)}
    queue = deque([start])
    while queue:
        tile = queue.popleft()
        distance, first_action = routes[tile]
        for action, neighbor in GRAPH[tile]:
            if neighbor not in routes:
                routes[neighbor] = (distance + 1, action if distance == 0 else first_action)
                queue.append(neighbor)
    return routes


ROUTES = {tile: routes_from(tile) for tile in TILES}
STARTS = {tile: tuple(ghost for ghost, (distance, _) in routes.items() if distance >= 7)
          for tile, routes in ROUTES.items()}
MOVES = {tile: dict(edges) for tile, edges in GRAPH.items()}


def corridor_risk(player, neighbor, ghost, power_pellets):
    """Return the risk tier and distance from the ghost on the first tile."""
    gap = ROUTES[neighbor][ghost][0]
    if gap <= 1 and (gap == 0 or neighbor not in power_pellets):
        return 2, gap
    current = ROUTES[player][ghost][0]
    risk = int(gap <= 2 and gap < current and neighbor not in power_pellets)
    previous, tile = player, neighbor
    for depth in range(1, 4):
        if tile in power_pellets:
            break
        distance = ROUTES[tile][ghost][0]
        onward = [target for _, target in GRAPH[tile] if target != previous]
        if ((depth > 1 and distance <= depth)
                or (not onward and distance <= 2 * depth + 2)):
            risk = 1
        if len(onward) != 1:
            break
        previous, tile = tile, onward[0]
    return risk, gap


def observation(pacman, ghost, pellets, power_pellets, fruit, powered):
    routes = ROUTES[pacman]
    nearest = next((first for tile, (_, first) in routes.items()
                    if tile != pacman and tile in pellets), -1)
    moves = MOVES[pacman]
    choices = tuple(moves)
    danger = caution = 0
    if ghost is not None and not powered and routes[ghost][0] <= 10:
        risks = {action: corridor_risk(pacman, neighbor, ghost, power_pellets)
                 for action, neighbor in moves.items()}
        danger = sum(1 << action for action, (risk, _) in risks.items() if risk == 2)
        caution = sum(1 << action for action, (risk, _) in risks.items() if risk == 1)
        best = min((risk, -gap if risk else 0) for risk, gap in risks.values())
        choices = tuple(action for action, (risk, gap) in risks.items()
                        if (risk, -gap if risk else 0) == best)

    bonus, direction = 0, -1
    if powered and ghost is not None and 0 < routes[ghost][0] <= 8:
        bonus, direction = EDIBLE_GHOST, routes[ghost][1]
    elif fruit and 0 < routes[FRUIT_TILE][0] <= 12:
        bonus, direction = FRUIT, routes[FRUIT_TILE][1]
    elif power_pellets:
        target = min(power_pellets, key=routes.__getitem__)
        distance, first = routes[target]
        if 0 < distance <= 8:
            bonus, direction = POWER, first
    fallback = direction if direction in choices else nearest
    return f"{VALID[pacman]}:{danger}:{caution}:{nearest}:{bonus}:{direction}", fallback, choices


class Maze:
    def __init__(self, seed):
        rng = Random(seed)
        self.pacman = rng.choice(TILES)
        self.ghost = rng.choice(STARTS[self.pacman])
        self.pellets = set(TILES) - {self.pacman}
        self.power_pellets = POWER_TILES & self.pellets
        self.power_turns = 0
        self.fruit_turns = 0
        self.power_eaten = self.fruit_eaten = self.ghosts_eaten = self.dots_eaten = 0
        self.score = 0
        self.steps = 0
        self.outcome = None

    def actions(self):
        return MOVES[self.pacman]

    def observe(self):
        return observation(self.pacman, self.ghost, self.pellets, self.power_pellets,
                           self.fruit_turns > 0, self.power_turns > 0)

    def step(self, action):
        if self.outcome is not None or action not in self.actions():
            raise ValueError("Choose a valid move in an unfinished episode.")
        self.pacman = self.actions()[action]
        self.steps += 1
        points = 0
        if self.pacman in self.pellets:
            self.pellets.remove(self.pacman)
            self.dots_eaten += 1
            points = 10
            if self.pacman in self.power_pellets:
                self.power_pellets.remove(self.pacman)
                self.power_eaten += 1
                self.power_turns = POWER_TURNS
                points = 50
            if self.dots_eaten in (12, 30):
                self.fruit_turns = FRUIT_TURNS
        if self.fruit_turns and self.pacman == FRUIT_TILE:
            self.fruit_turns = 0
            self.fruit_eaten += 1
            points += 100
        if self.ghost is not None and self.steps % 2 == 0 and self.pacman != self.ghost:
            if self.power_turns:
                self.ghost = max(MOVES[self.ghost].values(),
                                 key=lambda tile: ROUTES[self.pacman][tile][0])
            else:
                _, chase = ROUTES[self.ghost][self.pacman]
                self.ghost = MOVES[self.ghost][chase]
        if self.pacman == self.ghost:
            if self.power_turns:
                self.ghost = None
                self.ghosts_eaten += 1
                points += 200
            else:
                self.outcome = "caught"
        if self.power_turns:
            self.power_turns -= 1
            if not self.power_turns and self.ghost is None:
                self.ghost = STARTS[self.pacman][-1]
        if self.fruit_turns:
            self.fruit_turns -= 1
        if self.outcome is None and not self.pellets:
            self.outcome = "cleared"
        if self.outcome is None and self.steps == STEP_LIMIT:
            self.outcome = "timeout"
        self.score += points
        return points / 10 - 0.02 - CAUGHT_PENALTY * (self.outcome == "caught")


def greedy(table, state, fallback, actions):
    values = table.get(state)
    if values is None:
        return fallback if fallback in actions else next(iter(actions))
    return max(actions, key=values.__getitem__)


def train(seed, episodes=10_000):
    rng = Random(seed)
    table = {}
    returns = []
    for episode in range(episodes):
        maze = Maze(rng.randrange(1_000_000_000))
        epsilon = max(0.05, 1 - episode / (episodes * 0.8))
        state, fallback, choices = maze.observe()
        total = 0
        while maze.outcome is None:
            actions = maze.actions()
            action = (rng.choice(tuple(actions)) if rng.random() < epsilon
                      else greedy(table, state, fallback, choices))
            reward = maze.step(action)
            total += reward
            target = reward
            if maze.outcome is None:
                next_state, next_fallback, next_choices = maze.observe()
                future = table.get(next_state, (0, 0, 0, 0))
                target += 0.95 * max(future[move] for move in next_choices)
            values = table.setdefault(state, [0.0] * len(ACTIONS))
            values[action] += 0.15 * (target - values[action])
            if maze.outcome is None:
                state, fallback, choices = next_state, next_fallback, next_choices
        returns.append(total)
    return table, returns


def evaluate(table, seeds):
    """Use None for the random policy. Return per-episode records."""
    results = []
    for seed in seeds:
        maze = Maze(seed)
        rng = Random(seed ^ 0xA15EED)
        while maze.outcome is None:
            actions = maze.actions()
            if table is None:
                action = rng.choice(tuple(actions))
            else:
                action = greedy(table, *maze.observe())
            maze.step(action)
        results.append({
            "seed": seed,
            "pellets": maze.dots_eaten,
            "score": maze.score,
            "power": maze.power_eaten,
            "fruit": maze.fruit_eaten,
            "ghosts": maze.ghosts_eaten,
            "outcome": maze.outcome,
            "steps": maze.steps,
        })
    return results


def export_policy(table, path):
    Path(path).write_text(
        json.dumps({"version": 3, "actions": ACTIONS, "q": table}, allow_nan=False),
        encoding="utf-8",
    )


def self_check():
    assert all(len(routes) == len(TILES) for routes in ROUTES.values())
    maze = Maze(0)
    maze.pacman, maze.ghost, maze.pellets = (1, 1), (3, 1), {(2, 1)}
    maze.power_pellets.clear()
    assert maze.observe() == ("6:2:4:1:0:-1", 1, (2,))
    assert greedy({"6:2:4:1:0:-1": [0, 100, 0, 0]}, *maze.observe()) == 2
    assert maze.step(1) == 0.98 and maze.outcome == "cleared"
    maze = Maze(0)
    maze.pacman, maze.ghost = (1, 1), (2, 1)
    assert abs(maze.step(1) - -24.02) < 1e-9 and maze.outcome == "caught"
    maze = Maze(0)
    maze.pacman, maze.ghost = (2, 1), (1, 1)
    maze.pellets.add((1, 1))
    maze.power_pellets.add((1, 1))
    assert maze.step(3) == 24.98 and maze.score == 250
    assert maze.power_eaten == maze.ghosts_eaten == 1 and maze.ghost is None
    maze.pacman, maze.fruit_turns = (6, 5), 1
    assert maze.observe()[0].endswith(":2:3")
    maze.step(3)
    assert maze.fruit_eaten == 1 and not maze.fruit_turns
    maze.power_turns = 1
    maze.step(1)
    assert maze.ghost is not None and not maze.power_turns
    maze = Maze(0)
    maze.pacman, maze.ghost, maze.power_turns = (1, 1), (3, 1), 3
    assert maze.observe()[0].endswith(":3:1")
    maze = Maze(0)
    maze.pacman, maze.ghost, maze.dots_eaten = (1, 1), (9, 9), 11
    maze.pellets.add((2, 1))
    maze.step(1)
    assert maze.fruit_turns == FRUIT_TURNS - 1
    maze.fruit_turns = 1
    maze.step(3)
    assert not maze.fruit_turns and not maze.fruit_eaten
    maze = Maze(0)
    maze.steps = STEP_LIMIT - 1
    maze.step(next(iter(maze.actions())))
    assert maze.outcome == "timeout"
    table, _ = train(0, episodes=10)
    assert table and evaluate(table, [1_000_000_000]) == evaluate(table, [1_000_000_000])


if __name__ == "__main__":
    self_check()
