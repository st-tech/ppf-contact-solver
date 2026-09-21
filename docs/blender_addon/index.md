# 💡 Overview

[ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver) is a GPU-accelerated contact simulation engine developed by
[ZOZO, Inc](https://corp.zozo.com/en/). The Blender 5.0+ add-on
documented here is one front-end that ships with the engine: you
model the scene in Blender, assign material groups and pins, and the
add-on runs the solve -- on a remote host, or on the machine Blender
runs on -- and fetches the animation back.

The engine runs on NVIDIA GPUs through CUDA, on AMD GPUs through ROCm,
on Apple silicon through Metal, and on any CPU through a portable
SIMD build. Which of those a run uses is a per-connection choice; see
{ref}`Choosing the build <choosing-the-build>`.

If you are new, start with [Getting Started](getting_started/index.md). If you want to
wire up a specific backend, jump to [Connections](connections/index.md). If you
already have a connection and want to drive the solver day-to-day, go to
[Workflow](workflow/index.md). For example clips, see the [Gallery](gallery.md).
