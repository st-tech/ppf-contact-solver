# Git Hooks

This directory contains git hooks for the ZOZO's Contact Solver project.

## Pre-commit Hook

The pre-commit hook automatically clears output cells from Jupyter notebooks in the `examples/` directory before committing. This keeps the repository clean and avoids committing unnecessary output data.

## Installation

### Automatic Installation

Run the install script from the repository root:

```bash
./hooks/install.sh
```

### Manual Installation

Copy the hooks to your local `.git/hooks` directory:

```bash
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Requirements

- `python3` on `PATH`
- The hook delegates to `tools/clear_notebook_outputs.py`, which only uses
  the Python standard library (no `jupyter` / `nbconvert` install required).

## Behavior

When you commit changes to any `.ipynb` file in the `examples/` directory:
1. The hook automatically runs before the commit
2. All output cells are cleared from the notebooks
3. The cleaned notebooks are re-staged
4. The commit proceeds with cleaned notebooks

## Bypassing the Hook

If you need to commit notebooks with outputs (not recommended), you can bypass the hook using:

```bash
git commit --no-verify
```

## Troubleshooting

If the hook isn't working:
1. Ensure the hook file is executable: `chmod +x .git/hooks/pre-commit`
2. Verify `python3` is on `PATH`: `python3 --version`
3. Confirm the script is present: `ls tools/clear_notebook_outputs.py`