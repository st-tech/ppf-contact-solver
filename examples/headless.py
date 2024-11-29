from frontend import App
import os

app = App("headless").clear()

V, F = app.mesh.square(res=64)
app.asset.add.tri("sheet", V, F)

V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)
app.asset.add.tri("sphere", V, F)

scene = app.scene.create("five-curtains")

space = 0.25
for i in range(5):
    obj = scene.add("sheet")
    obj.at(i * space, 0, 0).rotate(90, "y")
    obj.direction([0, 1, 0], [0, 0, 1])
    obj.pin(obj.grab([0, 1, 0]))

scene.add("sphere").at(-1, 0, 0).pin().move_by([8, 0, 0], 5)
fixed = scene.build()

param = app.session.param()
param.set("friction", 0.0)
param.set("dt", 0.01)
param.set("min-newton-steps", 8)
param.set("frames", 60)

session = app.session.create("dt-001-newton-8").init(fixed).start(param, blocking=True)
assert os.path.exists(os.path.join(session.output_path(), "vert_60.bin"))

