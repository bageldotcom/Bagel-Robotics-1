from itertools import combinations
import numpy as np
import pytest
from vpp2.critical_windows.selection import critical_packing, CoverageConflict


def test_contiguous_critical_interval_does_not_fall_into_greedy_overlap_trap():
    frames = np.arange(0, 97, 8)
    required = np.zeros(128, dtype=bool)
    required[20:60] = True
    chosen, capacity = critical_packing(frames, required, 4)
    # Greedy largest overlap chooses frame 24, then cannot cover frame 20.
    # An episode-level plan instead chooses contiguous windows 0 and 32.
    assert capacity == 4
    assert frames[chosen].tolist() == [0, 32]


def test_packing_matches_exhaustive_search_on_small_episodes():
    frames = np.arange(0, 57, 8)
    rng = np.random.default_rng(41)
    for _ in range(30):
        required = rng.random(88) < .08
        cap = int(rng.integers(1, 4))
        feasible = []
        for count in range(1, cap + 1):
            for ids in combinations(range(len(frames)), count):
                starts = frames[list(ids)]
                if (np.diff(starts) < 32).any():
                    continue
                covered = np.zeros(88, dtype=bool)
                critical_windows = 0
                for start in starts:
                    covered[start:start+32] = True
                    critical_windows += int(required[start:start+32].any())
                if (covered | ~required).all():
                    feasible.append((count, critical_windows))
        if not feasible:
            with pytest.raises(CoverageConflict):
                critical_packing(frames, required, cap)
        else:
            chosen, capacity = critical_packing(frames, required, cap)
            best_capacity = max(count for count, _ in feasible)
            min_critical = min(n for count, n in feasible if count == best_capacity)
            assert capacity == best_capacity
            assert len(chosen) == min_critical
            covered = np.zeros(88, dtype=bool)
            for start in frames[chosen]:
                covered[start:start+32] = True
            assert (covered | ~required).all()
