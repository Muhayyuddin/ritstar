#!/usr/bin/env python3
"""Run the full planner comparison across all demo scenarios.

Environments:
  2D: Maze, Narrow, Forest, Terrain, Bug Trap, Hyper-Dense
  3D: Spheres, Dense Labyrinth
  6D: Shelf, Cluttered, Tabletop  (if PyBullet available)

All 7 planners × 10 MC trials × 150 iterations per planner.
"""

from rit_star.comparison import run_full_comparison

if __name__ == '__main__':
    results = run_full_comparison(
        n_trials=5,
        max_iterations=150,
        batch_size=100,
        base_seed=42,
        visualize=True,
    )
    print('\nDone. Results saved to results/ and visualization/plots/')
