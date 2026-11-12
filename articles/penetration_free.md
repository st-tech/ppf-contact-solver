# 🔒 Our Penetration-free Guarantee

We employ Continuous Collision Detection (CCD), **not Discrete Collision Detection (DCD)**. Our [cubic barrier](../README.md#-technical-materials) theoretically guarantees penetration-free contact by construction, not heuristics, **only when a solution is found.** A solution may not be found in extreme cases. In such cases, as well as with bugs, the solver may crash, stall, or take nearly forever to find a solution. **What we guarantee is that every step that succeeds is free of intersections.**

## ✅ How Do We Promise No Intersections?

An intersection checker confirms **zero penetration** after every successful run and every step in [GitHub Actions](../README.md#-github-actions), so no hidden intersections are left behind.

## 🚨 Intersection Errors Can Also Indicate an Impossible Setup

Examples include pinned panels driven through each other ([#116](https://github.com/st-tech/ppf-contact-solver/issues/116)) and cloth trapped in a self-intersecting character animation ([#110](https://github.com/st-tech/ppf-contact-solver/issues/110)). Stopping preserves the guarantee, while completing would silently accept penetration. Check the scene before reporting a bug.

If a cloth is pinched by an animated character or a shape-keyed collider (see [#160](https://github.com/st-tech/ppf-contact-solver/issues/160)), the solver may stop with "Newton solve made no progress (over-constrained configuration)". In that case, enable "Soft Constraint" on the group of the collider. The collider then gives way slightly where it presses on the cloth, and the scene can run to the end.

<img src="https://github.com/user-attachments/assets/f9053c49-d2a1-4117-962c-35455882e011" alt="Apply Soft Constraints" width="500">

If the cloth has pins, changing the pin type to "Pull" can also fix the failure. If you have any pins, please turn them all to "Pull" pins.

<img src="https://github.com/user-attachments/assets/3b9a9dd3-d585-48ff-8aa1-cf7d3313a5c0" alt="Pull pin type" width="280">
