# 👋 Getting Started

[ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver) is a
GPU-accelerated contact simulation engine. The Blender 5.0+ add-on is one
front end that ships with it and turns Blender into an interactive editor for
the solver. You model in Blender and assign material groups, pins, and
colliders. The add-on uploads the scene to a `ppf-cts-server` instance, runs
the simulation, and fetches the resulting animation back so you can scrub it on
the timeline.

By the end of this chapter you will have the add-on installed, a connection
open, a cloth sheet simulated, and the animation playing in the viewport.

```{toctree}
:maxdepth: 1
:hidden:

install
tour
first_simulation
```

Read the pages in this order:

1. [Install](install.md) adds the add-on to Blender.
2. [Tour of the UI](tour.md) names each panel in the sidebar and what it holds.
3. [Your First Simulation](first_simulation.md) builds a cloth-over-sphere
   scene and runs it from start to finish.

## Where to Go Next

- **[Connections](../connections/index.md)**: set up the backend that matches
  your environment (a native type on this machine, SSH, or Docker), pick which
  build of the solver it runs, and switch between setups in one click with
  connection profiles.
- **[Workflow](../workflow/index.md)**: material parameters, pin operations,
  keyframed scene parameters, invisible colliders, snap and merge, and the full
  lifecycle from **Transfer** through **Fetch**.
- **[Blender Python API](../integrations/python_api.md)**: automate what the
  sidebar does from Blender's scripting editor.
- **[Troubleshooting](../troubleshooting.md)**: problems grouped by where they
  show up, from install and connection through transfer, run, and fetch.
- **[Video Tutorials](../tutorial.md)** and **[Gallery](../gallery.md)**:
  recorded walkthroughs, and example clips with the files behind them.
