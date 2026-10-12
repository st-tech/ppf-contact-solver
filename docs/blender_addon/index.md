# 💡 Overview

[ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver) is a
GPU-accelerated contact simulation engine developed by
[ZOZO, Inc](https://corp.zozo.com/en/). It has two front-ends: the Blender
5.0+ add-on documented here, and a JupyterLab notebook interface (see the
[JupyterLab Python API](../jupyterlab_api/index.rst)).

With the add-on you model the scene in Blender, assign objects to
[groups](workflow/scene/object_groups.md#group-types) (**Solid**, **Shell**,
**Rod**, **PDRD** for rigid bodies, **Sand** for granular bodies, or
**Static**), and add pins and colliders. The add-on runs the solve on a
remote host or on the machine Blender runs on, and fetches the animation
back.

The engine runs on NVIDIA GPUs through CUDA, on Apple silicon through
Metal, and on any CPU through a portable SIMD build. The ROCm build for AMD
GPUs has not been run on AMD hardware; see
[Install](getting_started/install.md#prerequisites). You choose which build
a run uses with the **GPU** / **CPU** toggle (the **Compute Device**
setting), and with **GPU Backend** where a host has more than one GPU build;
see
{ref}`Choosing the build <choosing-the-build>`.

## Where to Go Next

- **New to the add-on:** [Getting Started](getting_started/index.md) installs
  the add-on, opens a connection, and runs a first cloth simulation.
- **Connecting to a solver:** [Connections](connections/index.md) covers each
  connection type. [Security](security.md) lists what to check before you
  point the add-on at a host you do not fully control.
- **Day-to-day work:** [Workflow](workflow/index.md) covers groups,
  parameters, pins, colliders, and the **Transfer**, **Run**, and
  **Fetch All Animation** loop.
  [JupyterLab](workflow/sim/jupyterlab.md) continues a transferred scene in
  a notebook on the solver host.
- **Scripts and agents:** the [Blender Python API](integrations/python_api.md)
  and the [MCP Server](integrations/mcp.md) drive the same operations from a
  script or an external agent.
- **Examples:** the [Gallery](gallery.md) and the
  [Video Tutorials](tutorial.md).
- **Problems and terms:** [Troubleshooting](troubleshooting.md) and the
  [Glossary](glossary.md).
- **Reference:** the
  [Blender Python API Reference](integrations/python_api_reference.rst) lists
  every public class and method, the
  [MCP Tool Reference](integrations/mcp_reference.rst) lists every tool, and
  the [JupyterLab Python API](../jupyterlab_api/index.rst) documents the
  notebook package and its parameters.
- **About this project:** [Reaching the Author](reaching_the_author.md) and
  [LLM Transparency](llm_transparency.md).
