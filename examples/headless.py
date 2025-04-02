# File: headless.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from frontend import App

app = App.create()

V, F = app.mesh.square(res=64, ex=[0, 0, 1], ey=[0, 1, 0])
app.asset.add.tri("sheet", V, F)

V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)
app.asset.add.tri("sphere", V, F)

scene = app.scene.create("five-curtains")

space = 0.25
for i in range(5):
    obj = scene.add("sheet")
    obj.at(i * space, 0, 0)
    obj.direction([0, 1, 0], [0, 0, 1])
    obj.pin(obj.grab([0, 1, 0]))

scene.add("sphere").at(-1, 0, 0).jitter().pin().move_by([8, 0, 0], 5)
fixed = scene.build()

param = (
    app.session.param()
    .set("friction", 0.0)
    .set("dt", 0.01)
    .set("min-newton-steps", 8)
    .set("frames", 60)
)

session = app.session.create(fixed).start(param, blocking=True)
session.export.animation().zip()

assert session.finished()
