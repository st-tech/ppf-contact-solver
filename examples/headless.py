# File: headless.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from frontend import App

app = App.create("headless")

V, F = app.mesh.square(res=64, ex=[0, 0, 1], ey=[0, 1, 0])
app.asset.add.tri("sheet", V, F)

V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)
app.asset.add.tri("sphere", V, F)

scene = app.scene.create()

space = 0.25
for i in range(5):
    obj = scene.add("sheet")
    obj.at(i * space, 0, 0)
    obj.direction([0, 1, 0], [0, 0, 1])
    obj.pin(obj.grab([0, 1, 0]))
    obj.param.set("strain-limit", 0.05)

scene.add("sphere").at(-1, 0, 0).jitter().pin().move_by([8, 0, 0], t_start=0.0, t_end=5)
scene = scene.build()

session = app.session.create(scene)
(
    session.param.set("dt", 0.01)
    .set("min-newton-steps", 8)
    .set("frames", 60)
)
session = session.build()
session.start(blocking=True)
session.export.animation().zip()

if app.ci:
    assert session.finished()