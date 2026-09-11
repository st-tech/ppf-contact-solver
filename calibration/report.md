# Fabric preset calibration report

How each bundled fabric preset behaves: how much it drapes, how much it droops under its own weight, and how easily it stretches. Each fabric is shown with simulated images and its parameters.

- **Drape**: a round fabric sheet hangs over a smaller disc. The drape coefficient is how much of its flat area the shadow still covers: a soft, drapey fabric collapses into deep folds and covers little (low %), while a stiff fabric stays spread out (high %). The values match published measurements for each fabric. The fold count is the number of waves around the rim (approximate).
- **Bending**: a 9 cm strip is held at one end and droops under gravity. The tip droop angle shows bending stiffness: a stiffer fabric droops less.
- **Stretch**: the young-mod and Poisson values describe how the fabric stretches in its plane. Silk and wool stretch more easily; denim and leather barely stretch.

## Summary

| Fabric | Drape DC % | Target % | Folds | Bend droop (deg) | bend | young-mod | Poisson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Silk | 33.0 | 20-40 | 10 | 78.7 | 1.42 | 500 | 0.4 |
| Flag | 26.4 | 20-40 | 10 | 82.3 | 0.83 | 1000 | 0.4 |
| Cotton | 68.2 | 60-76 | 10 | 64.2 | 4.3 | 5500 | 0.35 |
| Wool | 61.2 | 40-70 | 11 | 67.0 | 3.67 | 2000 | 0.4 |
| Denim | 85.9 | 70-90 | 8 | 45.9 | 10.0 | 10000 | 0.25 |
| Leather | 53.0 | 47-70 | 14 | 75.8 | 1.8 | 13000 | 0.4 |

## Silk

Drape coefficient **33.0%** &nbsp;|&nbsp; observed folds **10** &nbsp;|&nbsp; cantilever tip droop **78.7 deg** &nbsp;|&nbsp; bend `1.42`, young-mod `500`, Poisson `0.4`

![Silk drape, top view](report_images/Silk_top.png)

Looking straight down at the draped fabric. The shaded area is its shadow (the drape coefficient); the wavy edge is its folds.

![Silk drape, oblique view](report_images/Silk_oblique.png)

The same drape seen at an angle, showing the folds.

![Silk cantilever bend](report_images/Silk_side.png)

A strip held at the left, drooping under gravity. The blue line is its centerline; the angle it makes with horizontal (gray) is the droop.

## Flag

Drape coefficient **26.4%** &nbsp;|&nbsp; observed folds **10** &nbsp;|&nbsp; cantilever tip droop **82.3 deg** &nbsp;|&nbsp; bend `0.83`, young-mod `1000`, Poisson `0.4`

![Flag drape, top view](report_images/Flag_top.png)

Looking straight down at the draped fabric. The shaded area is its shadow (the drape coefficient); the wavy edge is its folds.

![Flag drape, oblique view](report_images/Flag_oblique.png)

The same drape seen at an angle, showing the folds.

![Flag cantilever bend](report_images/Flag_side.png)

A strip held at the left, drooping under gravity. The blue line is its centerline; the angle it makes with horizontal (gray) is the droop.

## Cotton

Drape coefficient **68.2%** &nbsp;|&nbsp; observed folds **10** &nbsp;|&nbsp; cantilever tip droop **64.2 deg** &nbsp;|&nbsp; bend `4.3`, young-mod `5500`, Poisson `0.35`

![Cotton drape, top view](report_images/Cotton_top.png)

Looking straight down at the draped fabric. The shaded area is its shadow (the drape coefficient); the wavy edge is its folds.

![Cotton drape, oblique view](report_images/Cotton_oblique.png)

The same drape seen at an angle, showing the folds.

![Cotton cantilever bend](report_images/Cotton_side.png)

A strip held at the left, drooping under gravity. The blue line is its centerline; the angle it makes with horizontal (gray) is the droop.

## Wool

Drape coefficient **61.2%** &nbsp;|&nbsp; observed folds **11** &nbsp;|&nbsp; cantilever tip droop **67.0 deg** &nbsp;|&nbsp; bend `3.67`, young-mod `2000`, Poisson `0.4`

![Wool drape, top view](report_images/Wool_top.png)

Looking straight down at the draped fabric. The shaded area is its shadow (the drape coefficient); the wavy edge is its folds.

![Wool drape, oblique view](report_images/Wool_oblique.png)

The same drape seen at an angle, showing the folds.

![Wool cantilever bend](report_images/Wool_side.png)

A strip held at the left, drooping under gravity. The blue line is its centerline; the angle it makes with horizontal (gray) is the droop.

## Denim

Drape coefficient **85.9%** &nbsp;|&nbsp; observed folds **8** &nbsp;|&nbsp; cantilever tip droop **45.9 deg** &nbsp;|&nbsp; bend `10.0`, young-mod `10000`, Poisson `0.25`

![Denim drape, top view](report_images/Denim_top.png)

Looking straight down at the draped fabric. The shaded area is its shadow (the drape coefficient); the wavy edge is its folds.

![Denim drape, oblique view](report_images/Denim_oblique.png)

The same drape seen at an angle, showing the folds.

![Denim cantilever bend](report_images/Denim_side.png)

A strip held at the left, drooping under gravity. The blue line is its centerline; the angle it makes with horizontal (gray) is the droop.

## Leather

Drape coefficient **53.0%** &nbsp;|&nbsp; observed folds **14** &nbsp;|&nbsp; cantilever tip droop **75.8 deg** &nbsp;|&nbsp; bend `1.8`, young-mod `13000`, Poisson `0.4`

![Leather drape, top view](report_images/Leather_top.png)

Looking straight down at the draped fabric. The shaded area is its shadow (the drape coefficient); the wavy edge is its folds.

![Leather drape, oblique view](report_images/Leather_oblique.png)

The same drape seen at an angle, showing the folds.

![Leather cantilever bend](report_images/Leather_side.png)

A strip held at the left, drooping under gravity. The blue line is its centerline; the angle it makes with horizontal (gray) is the droop.

---

Drape coefficients follow the Cusick method (BS 5058 / ISO 9073-9); cantilever bending follows ASTM D1388. Per-fabric reference values and citations are in the `calibration/` folder.
