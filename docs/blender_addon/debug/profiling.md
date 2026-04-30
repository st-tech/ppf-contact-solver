# Profiling

When the sidebar feels laggy or a viewport overlay is dragging redraws, the
in-addon draw-time profiler tells you which draw callbacks are eating the
frame. It wraps every add-on draw method so each call is timed, and
leaves everything else alone.

The profiler lives at `blender_addon/ui/perf.py`. You drive it from the
CLI at `blender_addon/debug/perf.py`, which ships Python snippets to
Blender over the same transport as `debug/main.py`.

## What It Instruments

| Category        | Target                                                                         |
| --------------- | ------------------------------------------------------------------------------ |
| Panels          | Every registered `Panel.draw` whose class lives in this add-on.                |
| UILists         | Every `UIList.draw_item`, `draw_filter`, and `filter_items` method.            |
| Viewport overlay | `POST_VIEW` and `POST_PIXEL` draw-handler callbacks in `ui/dynamics/overlay`. |

Each recorded entry stores call count, cumulative total, minimum, and
maximum time. Draw methods whose module is not part of this add-on are
left alone.

## CLI

```bash
python blender_addon/debug/perf.py enable           # install timing wrappers
python blender_addon/debug/perf.py sample           # enable + force redraws + report
python blender_addon/debug/perf.py report           # print current stats
python blender_addon/debug/perf.py report --json    # raw JSON for further analysis
python blender_addon/debug/perf.py reset            # clear collected stats
python blender_addon/debug/perf.py disable          # restore original draw methods
```

Options for `enable`:

| Flag            | Effect                                           |
| --------------- | ------------------------------------------------ |
| `--no-panels`   | Skip wrapping `Panel.draw` methods.              |
| `--no-uilists`  | Skip wrapping `UIList` methods.                  |
| `--no-overlay`  | Skip wrapping viewport overlay callbacks.        |

`sample` forces redraws via `bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',
iterations=N)` inside the same snippet that enables the profiler and
prints the report. It is the fastest way to get a first readout, but it
only covers panels and overlays that are **currently visible**. Panels in
a collapsed region, a closed sidebar, or a different workspace will report
zero samples.

Global options (same as `debug/main.py`):

| Option             | Default     | Description                       |
| ------------------ | ----------- | --------------------------------- |
| `--host HOST`      | `localhost` | Target host.                      |
| `--mcp-port PORT`  | `9633`      | MCP server port.                  |

## Recommended Workflow

1. **Enable.** Run `perf.py enable`. The profiler resets its stats and tags
   all areas for redraw, so the first samples land almost immediately.
2. **Reproduce.** Hover the mouse over the affected panels, resize the
   sidebar, scrub frames, toggle overlays, or anything else that would cause the
   draws you want to measure. The more you trigger the slow path, the more
   meaningful the numbers.
3. **Report.** Run `perf.py report`. Rows are sorted by cumulative total
   time.
4. **Disable.** Run `perf.py disable` when you are done. The wrappers are
   cheap but they do add a `time.perf_counter()` call per draw.

`reset` without `disable` is useful when you want to measure a specific
interaction: enable, idle, reset, interact, report.

## Reading the Output

Columns are fixed-width:

```text
name                                                         count  total(ms)  mean(ms)   max(ms)
----------------------------------------------------------------------------------------------------
Panel:DYNAMICS_PT_Groups.draw                                  42      318.14     7.575    18.402
Overlay:draw_overlay_callback                                 180       84.20     0.468     2.031
UIList:OBJECT_UL_PinOperationsList.draw_item                  612       31.44     0.051     0.330
```

Interpretation hints:

- **Sort order is cumulative total time.** A panel with a very high
  `count` but low `mean` is not necessarily slow, but it is being redrawn
  far more often than expected. This is usually a sign of over-aggressive
  `tag_redraw` calls somewhere (e.g. a timer that tags on every tick, or
  an operator that tags inside a loop).
- **Overlay entries appear by handler name**, prefixed with `Overlay:`.
  There are two hooks: one for geometry (POST_VIEW) and one for labels
  (POST_PIXEL text). If the labels hook dominates, the culprit is
  usually a large number of text draws per pin / operation.
- **`max` spikes** tell you about worst-case frames. A `mean` of 0.5 ms
  with a `max` of 20 ms means something occasionally re-builds geometry
  from scratch, which is worth profiling with `enable` and then looking at the
  cache version counters in `overlay_cache`.

:::{tip}
`UIList.draw_item` is called once per row every draw. If a list with 500
pins shows `count = 50000`, the list has been drawn 100 times. That is
expensive regardless of how fast each row is.
:::

## Related Pages

- [Hot Reload](hot_reload.md): the transport used by the profiler CLI.

:::{admonition} Under the hood
:class: toggle

**Overlay handler install**

Overlay handlers need a slightly different install than panels and
UILists because Blender stores a function pointer at registration time:
the profiler briefly unregisters the overlay handlers, swaps in the
timed versions, and re-registers them.

**Calling the profiler from your own script**

The CLI loads `ui/perf.py` dynamically because the add-on's top-level
package name depends on how Blender installed it (legacy vs extension).
The CLI scans loaded modules to find the add-on's top-level package
name, then imports `<pkg>.ui.perf` from there.

If you want to drive the profiler from a Python snippet that is not
routed through `debug/perf.py`, reuse the same pattern:

```python
import sys
import importlib

top = next(
    (n.split(".")[0] for n in sys.modules if n.endswith(".ui.dynamics.overlay")),
    None,
)
if top is None:
    raise RuntimeError("ppf-contact-solver addon does not appear to be loaded")

perf = importlib.import_module(top + ".ui.perf")
perf.enable()
perf.tag_redraw_all()
# ... trigger redraws ...
print(perf.report())
perf.disable()
```

This is also the right entry point if you are wiring a test or a
regression check that needs deterministic draw-time numbers.
:::
