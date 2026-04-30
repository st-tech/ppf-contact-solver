# 👋 Getting Started

[ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver) is a GPU-accelerated contact simulation engine; the
Blender 5.0+ add-on is one front-end that ships with it, turning Blender into
an interactive editor for the solver. You model in Blender, assign material
groups, pins, and colliders, and the add-on simulates remotely and fetches
the resulting animation back so you can scrub it on the timeline. By the end of this chapter you will have the add-on
installed, a connection open, a single cloth sheet simulated, and the
resulting animation playing in the viewport.

```{toctree}
:maxdepth: 1

install
tour
first_simulation
```

## Where to Go Next

- **[Connections](../connections/index.md)**: set up the backend that matches
  your environment (local, SSH, Docker, Windows native), and learn how
  connection profiles let you switch between them in one click.
- **[Workflow](../workflow/index.md)**: material parameters, pin operations,
  keyframed scene parameters, invisible colliders, snap-and-merge, and the
  full lifecycle from **Transfer** through **Fetch**.
- **[Blender Python API](../integrations/python_api.md)**: drive every operator on
  this page from a script or a Jupyter notebook instead of the sidebar.
