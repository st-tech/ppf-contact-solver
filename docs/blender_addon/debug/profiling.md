# Profiling

When the sidebar feels laggy or a viewport overlay slows redraws, the
in-add-on draw-time profiler shows which draw callbacks take the most time.
It wraps the add-on's panel draw methods, list draw methods, and the two
viewport-overlay callbacks so each call is timed, and leaves everything else
alone.

The profiler is `blender_addon/ui/perf.py`. You drive it from the CLI at
`blender_addon/debug/perf.py`, which ships Python snippets to Blender over the
same transport as `debug/main.py`. Blender must be running with the MCP server
or the reload server started; see [Hot Reload](hot_reload.md).

## What It Instruments

| Category         | Target                                                                          |
| ---------------- | ------------------------------------------------------------------------------- |
| Panels           | The `draw` method of each `Panel` class that this add-on defines.               |
| UILists          | The `draw_item`, `draw_filter`, and `filter_items` methods of each `UIList` class that this add-on defines. |
| Viewport overlay | The `POST_VIEW` callback `draw_overlay_callback` and the `POST_PIXEL` callback `_draw_overlay_labels` in `ui/dynamics/overlay.py`. |

Each recorded entry stores the call count and the cumulative, minimum, and
maximum time. Classes from other add-ons, operator `draw` methods (popup
dialogs), and panel headers are not wrapped.

## CLI

```bash
python blender_addon/debug/perf.py enable           # install timing wrappers
python blender_addon/debug/perf.py sample           # enable + force redraws + report
python blender_addon/debug/perf.py report           # print current stats
python blender_addon/debug/perf.py report --json    # raw JSON for further analysis
python blender_addon/debug/perf.py reset            # clear collected stats
python blender_addon/debug/perf.py disable          # restore original draw methods
```

`enable` installs the wrappers, clears the collected stats, tags every area
for redraw, and prints how many panels, list methods, and overlay callbacks it
wrapped. Its options:

| Flag            | Effect                                           |
| --------------- | ------------------------------------------------ |
| `--no-panels`   | Skip wrapping `Panel.draw` methods.              |
| `--no-uilists`  | Skip wrapping `UIList` methods.                  |
| `--no-overlay`  | Skip wrapping viewport overlay callbacks.        |

`report` prints the first 30 rows by default. `report --top N` changes the
count. `report --json` prints an object with `enabled` and `rows`, where each
row has `name`, `count`, `total_ms`, `mean_ms`, `max_ms`, and `min_ms`. The
text table omits `min_ms`.

`sample` forces redraws with `bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',
iterations=N)` inside the same snippet that enables the profiler and prints
the report. It is the fastest way to get a first readout. `--iterations N`
sets the redraw count (default 5) and `--top N` the row count (default 30).
It covers only the panels and overlays that are **currently visible**. Panels
in a collapsed region, a closed sidebar, or a different workspace do not appear
in the report. When nothing was recorded, the table has no rows and a line
under the header says there are no samples.

Global options go before the subcommand, as for `debug/main.py`:

| Option             | Default     | Description                       |
| ------------------ | ----------- | --------------------------------- |
| `--host HOST`      | `localhost` | Target host.                      |
| `--mcp-port PORT`  | `9633`      | MCP server port.                  |

## Recommended Workflow

1. **Enable.** Run `perf.py enable`. It clears the stats and tags all areas
   for redraw, so samples start to arrive at once.
2. **Reproduce.** Hover the mouse over the affected panels, resize the
   sidebar, scrub frames, toggle overlays, or do anything else that causes the
   draws you want to measure.
3. **Report.** Run `perf.py report`. Rows are sorted by cumulative total
   time.
4. **Disable.** Run `perf.py disable` when you are done. The wrappers make two
   `time.perf_counter()` calls per draw.

To measure one specific interaction without disabling the profiler, run
`enable`, leave Blender idle, run `reset`, perform the interaction, then run
`report`.

## Reading the Output

Columns are fixed-width. The values below are illustrative:

```text
name                                                          count  total(ms)  mean(ms)   max(ms)
----------------------------------------------------------------------------------------------------
Panel:DYNAMICS_PT_Groups.draw                                    42     318.14     7.575    18.402
Overlay:draw_overlay_callback                                   180      84.20     0.468     2.031
UIList:OBJECT_UL_PinOperationsList.draw_item                    612      31.44     0.051     0.330
```

Interpretation hints:

- **Rows are sorted by cumulative total time.** A callback with a high `count`
  and a low `mean` runs often rather than slowly. Look for code that calls
  `tag_redraw` repeatedly, for example a timer that tags on every tick or an
  operator that tags inside a loop.
- **Overlay entries appear by handler name**, prefixed with `Overlay:`. There
  are two hooks: `draw_overlay_callback` for geometry (`POST_VIEW`) and
  `_draw_overlay_labels` for text labels (`POST_PIXEL`). If the labels
  callback dominates, look at how many direction, operation, velocity,
  violation, and exemption labels are on screen.
- **`max` spikes** show the worst case. A `mean` of 0.5 ms with a `max` of
  20 ms means a call occasionally takes far longer than usual, for example
  when an overlay rebuilds its geometry. Check the cache keys in
  `_overlay_cache` (`ui/dynamics/overlay.py`), which decide when the overlay
  rebuilds.

:::{tip}
`UIList.draw_item` is timed once for each row a list draws, so the `count` of
a `draw_item` row grows with both the number of redraws and the number of rows
drawn per redraw. Use `mean` for the cost of one row and `total` for the whole
list.
:::

## Related Pages

- [Hot Reload](hot_reload.md): the transport used by the profiler CLI.

:::{admonition} Under the hood
:class: toggle

**Overlay handler install**

Overlay handlers need a different install from panels and UILists, because
Blender stores a function pointer when a draw handler is added. The profiler
unregisters the overlay handlers, replaces the two callbacks in the overlay
module with timed versions, and registers the handlers again.

**Calling the profiler from your own script**

The CLI loads `ui/perf.py` dynamically, because the add-on's top-level package
name depends on how Blender installed it (extension or legacy layout). It
scans the loaded modules for one that ends in `.ui.dynamics.overlay`, strips
that suffix to get the add-on's package name, and imports `<pkg>.ui.perf`.

To drive the profiler from a Python snippet that is not routed through
`debug/perf.py`, reuse the same pattern:

```python
import sys
import importlib

top = next(
    (n.removesuffix(".ui.dynamics.overlay") for n in sys.modules
     if n.endswith(".ui.dynamics.overlay")),
    None,
)
if top is None:
    raise RuntimeError("ppf_contact_solver addon does not appear to be loaded")

perf = importlib.import_module(top + ".ui.perf")
perf.enable()
perf.tag_redraw_all()
# ... trigger redraws ...
print(perf.report())
perf.disable()
```

`perf.report(top_n)` returns the text table (20 rows by default) and
`perf.report_json(top_n)` returns the same rows as a dictionary. Unlike the
CLI `enable`, `perf.enable()` does not clear the stats; call `perf.reset()`
for that.

This is also the entry point for a script or regression check that compares
draw-time measurements. The numbers are wall-clock samples from
`time.perf_counter()`, so expect run-to-run variation.
:::
