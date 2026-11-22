"""Genetic Algorithm for a short delivery route starting and ending at depot A."""
import random

# ---------------------------------------------------------------
# DISTANCE MATRIX (km) - symmetric, order: A, B, C, D, E, F, G
# ---------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DIST = [
    #  A  B  C  D  E  F  G
    [  0, 4, 8, 6, 7, 9, 5],  # A
    [  4, 0, 5, 7, 3, 8, 6],  # B
    [  8, 5, 0, 4, 6, 3, 7],  # C
    [  6, 7, 4, 0, 5, 6, 8],  # D
    [  7, 3, 6, 5, 0, 4, 6],  # E
    [  9, 8, 3, 6, 4, 0, 5],  # F
    [  5, 6, 7, 8, 6, 5, 0],  # G
]

# ---------------- GA parameters ----------------
SEED = 42
POP_SIZE = 30
GENERATIONS = 100          # at least 50
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
ELITE_COUNT = 2

DELIVERIES = list(range(1, len(LOCATIONS)))   # indices of B..G (A = 0)


def route_distance(chrom):
    """Total km for A -> chromosome -> A (includes the return to depot)."""
    path = [0] + chrom + [0]
    return sum(DIST[path[i]][path[i + 1]] for i in range(len(path) - 1))


def is_valid(chrom):
    return sorted(chrom) == DELIVERIES


def create_population(size):
    """Random permutations of B-G."""
    pop = []
    for _ in range(size):
        chrom = DELIVERIES[:]
        random.shuffle(chrom)
        pop.append(chrom)
    return pop


def tournament_select(pop):
    """Pick TOURNAMENT_SIZE random routes; the shortest wins."""
    contenders = random.sample(pop, TOURNAMENT_SIZE)
    return min(contenders, key=route_distance)


def order_crossover(p1, p2):
    """OX: copy a slice from p1, fill the rest in p2's order (no repeats/gaps)."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    remaining = [g for g in p2 if g not in child[a:b + 1]]
    it = iter(remaining)
    for i in range(n):
        if child[i] is None:
            child[i] = next(it)
    return child


def swap_mutation(chrom):
    """Swap two randomly chosen positions."""
    i, j = random.sample(range(len(chrom)), 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def run_ga():
    random.seed(SEED)
    pop = create_population(POP_SIZE)
    best = min(pop, key=route_distance)
    history = []

    for gen in range(1, GENERATIONS + 1):
        pop.sort(key=route_distance)
        new_pop = [c[:] for c in pop[:ELITE_COUNT]]      # elitism

        while len(new_pop) < POP_SIZE:
            p1, p2 = tournament_select(pop), tournament_select(pop)
            if random.random() < CROSSOVER_RATE:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            if random.random() < MUTATION_RATE:
                child = swap_mutation(child)
            assert is_valid(child)
            new_pop.append(child)

        pop = new_pop
        gen_best = min(pop, key=route_distance)
        if route_distance(gen_best) < route_distance(best):
            best = gen_best[:]
        history.append(route_distance(best))
        if gen == 1 or gen % 10 == 0:
            print(f"Generation {gen:3d}: best distance = {route_distance(best)} km")

    return best, history


if __name__ == "__main__":
    best, history = run_ga()
    names = ["A"] + [LOCATIONS[i] for i in best] + ["A"]
    print("\nBest route:", " -> ".join(names))
    print(f"Total distance: {route_distance(best)} km")
