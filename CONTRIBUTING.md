# Contributing

Thanks for considering it. This is a learning repo, so contributions that make
things *clearer* are as valuable as contributions that add features.

## Good first contributions

- **Fix an explanation that didn't land.** If a section confused you, it will
  confuse others. Open an issue describing where you got lost — that's genuinely
  useful even without a fix attached.
- **Finish a chapter.** Chapters 05–11 have complete outlines and starter code.
  Pick a section and write it out.
- **Add an exercise solution** in a `solutions/` notebook.
- **Catch a math error.** Please include the corrected derivation.

## Standards

Everything here is deliberately framework-free:

- **NumPy only** for implementations. `scikit-learn` may appear *only* for loading
  toy datasets or as a reference to check our results against — never as the
  implementation itself.
- **Derive, don't quote.** If you introduce a formula, show where it comes from.
- **Check your gradients.** Any new analytic gradient needs a test in
  `tests/test_gradients.py` comparing against `mlscratch.utils.numerical_gradient`.
  Relative error must be below `1e-6`.
- **Comment the non-obvious.** Explain *why* a line exists, not what it does.
  `# subtract the max so exp never overflows` is useful. `# add one to i` is not.
- **Set seeds.** Use `np.random.default_rng(seed)` so output is reproducible.

## Before opening a PR

```bash
pytest tests/ -v                       # all tests must pass

# Restart-and-run-all every notebook you touched, so outputs match the code
jupyter nbconvert --execute --inplace notebooks/your_notebook.ipynb
```

Keep notebook diffs small — commit executed outputs, but don't reshuffle cells
unnecessarily.

## Style

- Line length 100
- Type hints optional; clarity is not
- One idea per cell in notebooks; markdown cell explains, code cell demonstrates
