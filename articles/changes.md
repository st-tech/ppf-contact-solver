### 📝 Change History

---

This page records the full change history of the project, newest first.
For the latest highlights, see the [README](../README.md).

### 2026

- **(2026.06.30)** Added rigidbody support based on [Painless Differentiable Rotation Dynamics](https://dl.acm.org/doi/10.1145/3730944).
- **(2026.06.30)** Added sand support.
- **(2026.06.30)** Achieved 2x performance improvement [with help of feedback from @Hurleyworks](https://github.com/st-tech/ppf-contact-solver/discussions/86).
- **(2026.04.30)** Added Blender Add-on support. See the [documentation](https://st-tech.github.io/ppf-contact-solver).

### 2025

- **(2025.12.18)** Added native Windows standalone executable build support [(Video)](https://zozo.box.com/s/9rthkw122fyss5qxuf5mie9xywg7jzdz).
- **(2025.11.26)** Added [large-woven.ipynb](../examples/large-woven.ipynb) [(Video)](https://zozo.box.com/s/kc81msjfo4yw9eozn8i8bean0gbph0pj) to [large scale examples](../README.md#️-large-scale-examples).
- **(2025.11.12)** Added [five-twist.ipynb](../examples/five-twist.ipynb) [(Video)](https://zozo.box.com/s/36h8jpu5vcgc5t4xln2l68afj7izsx4h) and [large-five-twist.ipynb](../examples/large-five-twist.ipynb) [(Video)](https://zozo.box.com/s/v62q7cbfnpl3hufwwy2nmewyes2w1iw6) showcasing over 180M count. See [large scale examples](../README.md#️-large-scale-examples).
- **(2025.10.03)** Massive refactor of the codebase [(Markdown)](./refactor_202510.md). Note that this change includes breaking changes to our Python APIs.
- **(2025.08.09)** Added a hindsight note in [eigensystem analysis](./eigensys.md) to acknowledge prior work by [Poya et al. (2023)](https://romeric.github.io/).
- **(2025.05.01)** Simulation states now can be saved and loaded [(Video)](https://zozo.box.com/s/7v0exrbptvfli4o4z91pqtmz0tehdn62).
- **(2025.04.02)** Added 9 examples. See the [catalogue](../README.md#️-catalogue).
- **(2025.03.03)** Added a [budget table on AWS](../README.md#-budget-table-on-aws).
- **(2025.02.28)** Added a [reference branch and a Docker image of our TOG paper](../README.md#-technical-materials).
- **(2025.02.26)** Added Floating Point-Rounding Errors in ACCD in [hindsight](./hindsight.md).
- **(2025.02.07)** Updated the [trapped example](../examples/trapped.ipynb) [(Video)](https://zozo.box.com/s/lnnyeqrvm86rxnwyjxhojfj0jgm5nphn) with squishy balls.
- **(2025.01.08)** Added a [domino example](../examples/domino.ipynb) [(Video)](https://zozo.box.com/s/p5ksfqja1ew3c6vntco5zq6g0kgf7xoo).
- **(2025.01.05)** Added a [single twist example](../examples/twist.ipynb) [(Video)](https://zozo.box.com/s/4phoyyeertd2mcfv436kp2ojmo1x0eio).

### 2024

- **(2024.12.31)** Added full documentation for Python APIs, parameters, and log files [(GitHub Pages)](https://st-tech.github.io/ppf-contact-solver).
- **(2024.12.27)** Line search for strain limiting is improved [(Markdown)](./bug.md#new-strain-limiting-line-search).
- **(2024.12.23)** Added [(Bug Fixes and Updates)](./bug.md).
- **(2024.12.21)** Added a [house of cards example](../examples/cards.ipynb) [(Video)](https://zozo.box.com/s/7c114pua0107xkz4nc3bwfdzpkhgn1o9).
- **(2024.12.18)** Added a [frictional contact example](../examples/friction.ipynb): armadillo sliding on the slope [(Video)](https://zozo.box.com/s/15r5o7rrowwtbrsrjjpj35v8xt92ufhr).
- **(2024.12.18)** Added a [hindsight](./hindsight.md) noting that the tilt angle was not $30^\circ$, but rather $26.57^\circ$.
- **(2024.12.16)** Removed thrust dependencies to fix runtime errors for the driver version `560.94` [(Issue Link)](https://github.com/st-tech/ppf-contact-solver/issues/1).
