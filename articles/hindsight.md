## ️🧐 Hindsight and Follow-Ups

### Quadratic Barrier Function (Section 5.4)

In the paper we presented the following barrier as a quadratic energy counterpart:

```math
\begin{equation}
    \psi_{\mathrm{quad}}(g,\hat{g},\kappa) = \frac{\kappa}{2 \hat{g}} \left(g - \hat{g}\right)^2, \nonumber
\end{equation}
```
<div align="center">
<img src="../asset/image/hindsight/error.jpg" alt="equation" width="300">
</div>

However, soon after the publication, we realized that this was not the best counterpart since its curvature is a constant $\kappa / \hat{g}$.
Our elasticity-inclusive stiffness assumes that the curvature should be on the order of $\kappa$, implying that its magnitude does not align.
Since our cubic energy yields a curvature of $2 \kappa$ at $g = \hat{g}/2$, a more suitable counterpart would be:

```math
\color{red}
\begin{equation}
    \psi^{\mathrm{new}}_{\mathrm{quad}}(g,\hat{g},\kappa) = \kappa \left(g - \hat{g}\right)^2, \nonumber
\end{equation}
\color{black}
```
In hindsight, we discovered that with this change, the majority of artifacts arising from the use of quadratic barriers have improved, **but objectionable issues persist.**
We show an example and discuss why.

### 📉 Artifacts

When the above new quadratic barrier is used, visual artifacts may emerge **when contacts are lightly touched** as shown in Figure A.

<img src="../asset/image/hindsight/domino-artifacts.gif" alt="snag artifacts">

**Figure A:** Domino scene. Noticeable snags occur when one domino pushes the next ones.

### 🔄 The Sources of Artifacts

One of the most important differences between quadratic and cubic barriers is **how the curvature varies** from $g = \hat{g}$ to $g = 0$.
For our cubic barrier, the curvature starts from zero at $g = \hat{g}$ and gradually increases to $4\kappa$ at $g = 0$.

In contrast, the quadratic barrier produces its maximum curvature everywhere in $g < \hat{g}$.
In other words, our cubic barrier gradually stiffens the system as the gap shrinks to zero, while **the quadratic barrier maxes out the stiffness immediately when the barrier is turned on.**
This can be seen in Figure B.

> [!NOTE]
> You can think of this sort of like an ill-configured CPU fan controller, where the fan always runs at full throttle despite low CPU usage.
> Ideally, the fan should spin in accordance with the CPU temperature, much like our cubic energy where stiffness gradually increases. The quadratic energy, on the other hand, acts like this ill-configured controller; it gets the job done but is mostly overwhelming.

As a result, when $g \approx \hat{g}$ (that is, contacts are lightly touched), the conditioning of the system unnecessarily stiffens, leading to possible artifacts.

<img src="../asset/image/hindsight/search_dir.svg" alt="graph">

**Figure B:** Visualizing the transition of the magnitude of both our cubic barrier and a quadratic counterpart.