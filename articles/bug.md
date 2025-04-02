## Bug Fixes 🐞 and Updates 🔄

### New Strain Limiting Line Search

On 2024-Dec-27, we made minor improvements to the line search algorithm for strain limiting.
The new algorithm is laid out below.
Compared to the one presented in the supplementary PDF, this new algorithm searches for `t` (time of impact) with exact numerical precision; no parameters, such as the `eps` tolerance for checking convergence or the maximum loop count, are necessary.
This new approach converges quickly and exits the loop fast.

```c++
/* Apache v2.0 License */
float line_search_strain_limiting(
    const Mat3x2f &F0, // Deformation gradient at t = 0.0
    const Mat3x2f &F1, // Deformation gradient at t = 1.0
    float t,           // Maximal time of impact, such as t = 1.0
    float max_sigma )  // Maximal sigma, such as 1.01
{
    const Mat3x2f dF = F1 - F0;
    if (svd3x2(F0 + t * dF).S.max() >= max_sigma) {
        float upper_t = t;
        float lower_t = 0.0f;
        float window = upper_t - lower_t;
        while (true) {
            t = 0.5f * (upper_t + lower_t);
            float diff = svd3x2(F0 + t * dF).S.max() - max_sigma;
            if (diff < 0.0f) {
                lower_t = t;
            } else {
                upper_t = t;
            }
            float new_window = upper_t - lower_t;
            if (new_window == window) {
                break;
            } else {
                window = new_window;
            }
        }
        t = lower_t;
    }
    return t;
}
```

### BVH Construction

In the supplementary PDF, we mentioned that the BVH is reconstructed every 10 video frames; however, this is not what is implemented in the public code.

In this code, the BVH is continuously updated on the CPU side in the background without blocking the simulation. At the beginning of each step, we check if the BVH construction is finished, and if it is, we update the GPU buffer.

### Hang Example

On 2024-Dec-22, we encountered a situation where the PCG solver in our "hang" example `hang.ipynb` failed to converge, resulting in a simulation failure.

This issue occurred because the step size was set to `0.01`, which turned to be unreliably large for this setting.
We have now changed it to `0.001`.
This makes the system more diagonally dominant, leading to improved conditioning.

In this particular case, the strain-limiting constraint distance became very tight, which made the condition number challenging for the PCG solver to handle.

Note that despite the failure with a step size of `0.01`, this step size usually succeeds.
This is why we could not notice this issue before making our code public and testing many times.

We understand that this is a limitation: for some large step sizes, our method may fail.
One promising fix is to monitor the PCG, and if it fails, re-do the entire step with a smaller step size.
This option can be enabled by `param.set("enable_retry", True)`, but it is disabled by default.
We confirmed that with this option enabled, a larger step size of `0.1` can be chosen.

