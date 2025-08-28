## 💡 Tips for Obtaining Force Jacobians

### 🎓 For Scholars

**Hindsight (Aug 9 2025)**: After publication, we learned that the insights of the eigensystem analysis presented here were previously developed by [Poya et al.
(2023)](https://romeric.github.io/).
Readers who wish to mention this article should refer to the original work:

```bibtex
@article{poya2023variational,
  author = {Poya, Roman and Ortigosa, Rogelio and Gil, Antonio J.},
  title = {Variational schemes and mixed finite elements for large strain isotropic elasticity in principal stretches: Closed-form tangent eigensystems, convexity conditions, and stabilised elasticity},
  journal = {International Journal for Numerical Methods in Engineering},
  volume = {124},
  number = {16},
  pages = {3436-3493},
  keywords = {convexity conditions, large strain elasticity, mixed finite elements, principal stretches},
  doi = {https://doi.org/10.1002/nme.7254},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.7254},
  year = {2023}
}
```

We believe our article here remains valuable. For example, our treatment extends to shell elements with 3×2 deformation gradients. For comprehensive technical details, readers are referred to our supplementary materials in Sections E and F.9 of the [supplementary material](https://drive.google.com/file/d/1ptjFNVufPBV4-vb5UDh1yTgz8-esjaSF/view).

## 🏁 The Final Results

Since the materials below are lengthy, we provide a summary of the final results. If you're excited, go ahead and proceed reading:

- Any eigen-filtered force jacobian of isotropic energies expressed purely in terms of their singular values can all be computed in closed form, provided that $\frac{\partial \Psi}{\partial \sigma}$ and $\frac{\partial^2 \Psi}{\partial \sigma^2}$ are symbolically known.

- If the energy is expressed in terms of invariants, eigen-filtered force jacobian can be obtained in closed form by using only $\frac{\partial \Psi}{\partial I_k}$ and $\frac{\partial^2 \Psi}{\partial I_k^2}$, where $I_k$ are the Cauchy-Green invariants or those from Smith et al. [(1)](#1).

- Our analysis can be used to re-derive exactly the same system from Smith et al. [(1)](#1).

## 📖 What's the Story?

If you are solving it implicitly, you’ll probably need both the force and the force jacobian of a potential energy ⚡ with respect to 🎢 strain.
For example, let $\Psi(F)$ denote a hyper-elasticity potential energy with respect to a strain tensor $F$.
$F$ is a $3 \times 3$ matrix for solids and a $3 \times 2$ matrix for shells. It is alternatively called the deformation gradient.
Force derivatives refer to $\frac{\partial^2 \Psi}{\partial f_i \partial f_j}$, where **$f_i$ is a single entry of matrix $F$**.

## 🚧 What's the Problem?

We could manage 😖 to calculate the force $\frac{\partial \Psi}{\partial f_i}$, but the force jacobian $\frac{\partial^2 \Psi}{\partial f_i \partial f_j}$ is so hard 😵‍💫 that it drives us crazy! 🤯
But what a bummer! 😰 The reality is **cruel 💀**; what we actually need is the 🔍 **eigen-decomposed force jacobian**, not simply the force jacobian! 😭
They are necessary to project them onto the semi-positive space, ensuring that the Newton's method does descend the energy to be minimized. 🤔
A brute-force solution using numerical factorization is too 🐌 slow (it's a dense $9 \times 9$ matrix!).
Symbolic math software 💥 explodes.
🚧 Dead end 🚧.

> [!NOTE]
> Some readers may instead opt for position-based dynamics or projective dynamics, which do not require force jacobian.
> However, these first-order convergent methods are not suitable for highly stiff materials or high-resolution meshes due to their slow convergence 🐢.
> If you’re skeptical, try it yourself.
> They are best suited for real-time applications, such as 🎮 video games.

## 🤔 Why Not Just Use Past Literature?

We know that there is excellent literature out there 📚 but it has ⛔ limitations.
The most popular one [(1)](#1) heavily depends on invariants, but not all isotropic energies can be clearly expressed with them, or at least not without considerable human effort.
For example, how can we write a flavor of [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model) energy $\sum_k \left( \lambda_1^{0.5^k} + \lambda_2^{0.5^k} + \lambda_3^{0.5^k} - 3 \right)$ using invariants?

> [!NOTE]
> A practical example where our singular-value eigenanalysis is essential is the isotropic strain-limiting energies, which are discussed in the supplementary.

Other method [(2)](#2) can handle such cases but have singularities when two stretches are the same along principal axes.
This means that even a rest pose 🛟 is considered a singular case.
Technically, our method is mathematically a variant of Zhu [(3)](#3); though, we eliminate (not all) the singularities they have and offer a more explicit solution.
A notable difference is that ours has the ability to re-derive the eigen system revealed by Smith et al. [(1)](#1).
This is only possible with our formulation, not with Zhu [(3)](#3).

<a id="1">[1]</a> Breannan Smith, Fernando De Goes, and Theodore Kim. 2019. Analytic Eigensystems for Isotropic Distortion Energies. ACM Trans. Graph. <https://doi.org/10.1145/3241041>

<a id="2">[2]</a> Alexey Stomakhin, Russell Howes, Craig Schroeder, and Joseph M. Teran. 2012. Energetically consistent invertible elasticity. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '12).

<a id="3">[3]</a> Yufeng Zhu. 2021. Eigen Space of Mesh Distortion Energy Hessian. <https://arxiv.org/abs/2103.08141>

## 🎯 Our Goals

Without further ado, let us pin 📌 our objectives:

- We just need $\frac{\partial \Psi}{\partial f_i}$ and the eigen-filtered $\frac{\partial^2 \Psi}{\partial f_i \partial f_j}$; nothing else.
- We only use singular values, not invariants.
- It has to be both reasonably fast 🚀 and simple ✏️.

> [!TIP]
> Once you obtain $\frac{\partial \Psi}{\partial f_i}$ and $\frac{\partial^2 \Psi}{\partial f_i \partial f_j}$; we can apply the following chain rule
>
> ```math
> \begin{align}
>   \frac{\partial \Psi}{\partial x} = \sum_j \frac{\partial f_j }{\partial x} \frac{\partial \Psi}{\partial f_j}, \nonumber \\
>   \frac{\partial^2 \Psi}{\partial x^2} = \sum_{i,j} \frac{\partial^2 \Psi}{\partial f_i \partial f_j} \left(\frac{\partial f_i}{\partial x}\right)\left(\frac{\partial f_j}{\partial x}\right)^T, \nonumber
> \end{align}
> ```
>
> to convert them to $\frac{\partial \Psi}{\partial x}$ and $\frac{\partial^2 \Psi}{\partial x^2}$, where $x$ is a element vertex of interest.
> Once converted, both can be directly plugged into the Newton's method.
> Just to be sure; the second chain rule only holds when $\frac{\partial^2 f_i}{\partial x^2} = 0$.
> For linear elements, $F$ is linear in $x$. $\frac{\partial f_i}{\partial x}$ is a simple matter to compute, if you know how to get $F$ from $x$.

## 🔍 Eigen-decomposed Force Jacobians

If you are only interested in **isotropic** squishables 🎈 (e.g., their stretches aren't biased in any direction), we have a good solution 🎁 for it.

### 🪵 Volumetric Elasticity Energies

For 3D elasticity strain $F_{\mathrm{3D}}$ is a $3 \times 3$ matrix. First we realize that isotropic distortion energies are written as a function of singular values of $F$ such that $\Psi_{\mathrm{3D}}(\sigma_1,\sigma_2,\sigma_3)$, where $\sigma_{1 \cdots 3}$ are the singular values of $F$.
Let us list some popular ones (with Lamé and model-specific parameters substituted):

| Model | $\Psi_{\mathrm{3D}}(\sigma_1,\sigma_2,\sigma_3)$ |
|-------|------------|
| **ARAP** | $\sum_{j=1}^{3} (\sigma_j - 1)^2$ |
| **Symmetric Dirichlet** | $\sum_{j=1}^{3} \left(\sigma_j^2 + \frac{1}{\sigma_j^2}\right)$ |
| **MIPS** | $\sum_{j=1}^{3} \frac{\sigma_j^2}{\sigma_1 \sigma_2 \sigma_3}$ |
| **Ogden** | $\sum_{k=0}^{4} \left(\sigma_1^{0.5^k} + \sigma_2^{0.5^k} + \sigma_3^{0.5^k} - 3\right)$ |
| **Yeoh** | $\sum_{k=1}^{3} \left(\sigma_1^2 + \sigma_2^2 + \sigma_3^2 - 3\right)^{k}$ |

$\sigma_{1 \cdots 3}$ are obtained via a numerical SVD of $F$ at a reasonable time.
It's just a $3 \times 3$ matrix.
You may find that our closed-formed SVD routine (Apache v2.0) below is useful.

```
/* Apache v2.0 License */
template <unsigned R, unsigned C>
Svd<R, C> svd(const Mat<R, C> &F) {
    Mat<C, C> V;
    Vec<C> lambda;
    solve_symm_eigen<C>(F.transpose() * F, lambda, V);
    for (int i = 0; i < C; ++i) {
        lambda[i] = sqrtf(fmax(0.0f,lambda[i]));
    }
    Mat<R, C> U = F * V;
    for (int i = 0; i < U.cols(); i++) {
        U.col(i).normalize();
    }
    /* Do an inversion-aware flip if needed */
    return {U, lambda, V.transpose()};
}
```

This code can also handle non-square $F$, which is helpful for shell elements that we will cover later.
At the line ```solve_symm_eigen()``` we can use a closed-form solution.

> [!NOTE]
> You may argue that the eigen decomposition of symmetric $3 \times 3$ matrices is expensive.
> Yes, an iterative method like the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) is expensive.
> But for such a small matrix size, a direct solver is possible.
> It may sound overwhelming, but aside from finding the three roots of a cubic equation, everything else is trivial.
> I've written our own closed-form $3 \times 3$ eigen decomposition [(Code Link)](../eigsys/eig-hpp), so go ahead and use it!

Let's get our hands dirty 👐. We’ll first write down a set of matrices and scalars.

```math
\begin{align}
    Q_1 = \frac{1}{\sqrt{2}} U_{3 \times 3} \begin{bmatrix}
        0 & 1 & 0 \\
        -1 & 0 & 0 \\
        0 & 0 & 0
    \end{bmatrix} V_{3 \times 3}^T, \hspace{2mm}
    Q_2 = \frac{1}{\sqrt{2}} U_{3 \times 3} \begin{bmatrix}
        0 & 0 & 1 \\
        0 & 0 & 0 \\
        -1 & 0 & 0
    \end{bmatrix} V_{3 \times 3}^T, \nonumber \\
    Q_3 = \frac{1}{\sqrt{2}} U_{3 \times 3} \begin{bmatrix}
        0 & 0 & 0 \\
        0 & 0 & 1 \\
        0 & -1 & 0
    \end{bmatrix} V_{3 \times 3}^T, \hspace{2mm}
    Q_4 = \frac{1}{\sqrt{2}} U_{3 \times 3} \begin{bmatrix}
        0 & 1 & 0 \\
        1 & 0 & 0 \\
        0 & 0 & 0
    \end{bmatrix} V_{3 \times 3}^T, \nonumber \\
    Q_5 = \frac{1}{\sqrt{2}} U_{3 \times 3} \begin{bmatrix}
        0 & 0 & 1 \\
        0 & 0 & 0 \\
        1 & 0 & 0
    \end{bmatrix} V_{3 \times 3}^T, \hspace{2mm}
    Q_6 = \frac{1}{\sqrt{2}} U_{3 \times 3} \begin{bmatrix}
        0 & 0 & 0 \\
        0 & 0 & 1 \\
        0 & 1 & 0
    \end{bmatrix} V_{3 \times 3}^T. \nonumber
\end{align}
```

$U_{3 \times 3}$ and $V_{3 \times 3}^T$ are rotation matrices obtained by the above SVD call of $F$.
Next, write down the following $3 \times 3$ symmetric matrix:

```math
\begin{equation}
    H_{3 \times 3} = \begin{bmatrix}
    \displaystyle \frac{\partial^2}{\partial \sigma_1^2} & \displaystyle \frac{\partial^2}{\partial \sigma_1 \partial \sigma_2} & \displaystyle \frac{\partial^2}{\partial \sigma_1 \partial \sigma_3} \\
     & \displaystyle \frac{\partial^2}{\partial \sigma_2^2} & \displaystyle \frac{\partial^2}{\partial \sigma_2 \partial \sigma_3} \\
    \mathrm{Sym} & & \displaystyle \frac{\partial^2}{\partial \sigma_3^2}
    \end{bmatrix}\Psi_{\mathrm{3D}}. \nonumber
\end{equation}
```

Let $w_1$, $w_2$, and $w_3$ be a set of eigenvectors of $H_{3 \times 3}$, and $\beta_1$, $\beta_2$, and $\beta_3$ be the corresponding eigenvalues.
We can use a $3 \times 3$ numerical eigen solver to get them.
For a simple case where $H_{3 \times 3}$ is diagonal, such as in ARAP, this decomposition is not necessary.
Let us write down a few more things:

```math
\begin{align}
    \lambda_1 = \frac{1}{\sigma_1 + \sigma_2}\left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_1}+\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_2}\right), \hspace{3mm}
    \lambda_2 = \frac{1}{\sigma_1 + \sigma_3}\left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_1}+\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_3}\right), \nonumber \\
    \lambda_3 = \frac{1}{\sigma_2 + \sigma_3}\left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_2}+\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_3}\right), \hspace{3mm}
    \lambda_4 = \frac{1}{\sigma_1 - \sigma_2}\left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_1}-\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_2}\right), \nonumber \\
    \lambda_5 = \frac{1}{\sigma_1 - \sigma_3}\left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_1}-\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_3}\right), \hspace{3mm}
    \lambda_6 = \frac{1}{\sigma_2 - \sigma_3}\left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_2}-\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_3}\right), \nonumber 
\end{align}
```

```math
\begin{align}
    Q_7 = U_{3 \times 3} \textrm{diag}\left(w_1\right) V_{3 \times 3}^T, \nonumber \\
    Q_8 = U_{3 \times 3} \textrm{diag}\left(w_2\right) V_{3 \times 3}^T, \nonumber \\
    Q_9 = U_{3 \times 3} \textrm{diag}\left(w_3\right) V_{3 \times 3}^T, \nonumber \\
    \lambda_7 = \beta_1, \hspace{2mm}
    \lambda_8 = \beta_2, \hspace{2mm}
    \lambda_9 = \beta_3. \nonumber
\end{align}
```

All right, we have everything we need.
Make sure that we have 9 matrices $Q_{1 \cdots 9}$ and their corresponding scalars $\lambda_{1 \cdots 9}$ ready.

> [!NOTE]
> Some readers may notice that our results are consistent with [(1)](#1) in that the first six eigenmatrices are static, while the remaining three depend on the deformation. Indeed, our technique is closely related to [(1)](#1).
> $Q_{1 \cdots 3}$ are called twist modes, $Q_{4 \cdots 6}$ are called flip modes, and $Q_{7 \cdots 9}$ are called scaling modes.

Now we present results.
The first derivative (the force) is

```math
\begin{align}
    \frac{\partial \Psi_{\mathrm{3D}}}{\partial f_i} = \sum_k^3 \frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_k} \left( u_k \cdot \left( S_i v_k \right)\right), \nonumber
\end{align}
```

where $S_i$ is a $3 \times 3$ stencil matrix with a single entry of one at the same row and column as $f_i$, and zeros elsewhere.
Just to be clear, $F$ is a $3 \times 3$ matrix and thus has 9 entries. Therefore, $f_i$ loops from 1 to 9.
The loop order in the matrix view does not matter; even a random order will work.
$u_k$ and $v_k$ are vectors representing the $k$-th columns of the rotation matrices $U_{3 \times 3}$ and $V_{3 \times 3}$, respectively.
Finally, the eigen-filtered force jacobian are

```math
\begin{equation}
    \frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial f_i \partial f_j} = \sum_k^9 \max(\lambda_k,0) q_i q_j, \nonumber
\end{equation}
```

where $q_k$ is a single entry of $Q_k$ located at the same row and column as $f_k$.
Yes, it’s really that simple.
You don't believe it?
We've writen a Python code to numerically check its correctness. Check this out [(Code)](../eigsys/eigsys_3.py).

Okay, there's one more step.
When $\sigma_i = \sigma_j$, $\lambda_{4,5,6}$ result in division by zero.

> [!NOTE]
> Of course, if you use a symbolic package to simplify the symbolic expressions, singularities like $\sigma_i = \sigma_j$ may be removed. However, wouldn’t it be better if we could eliminate these singularities while keeping $\Psi_{\mathrm{3D}}$ abstracted?
> This way, you won’t have to manually translate the symbolic expressions into code.

But in this case, we can use the following approximation instead:

```math
\begin{equation}
   \frac{1}{\sigma_i - \sigma_j} \left( \frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_i} - \frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_j} \right) \approx \frac{1}{2}\left(\frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial \sigma_i^2} + \frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial \sigma_j^2}\right) - \frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial \sigma_i \partial \sigma_j}. \nonumber
\end{equation}
```

> [!NOTE]
> For this approximation to work, the energy definition must satisfy the Valanis-Landel hypothesis: $\Psi_{\mathrm{3D}}(\sigma_1, \sigma_2, \sigma_3)$ should be invariant to the order of $\sigma_1$, $\sigma_2$, and $\sigma_3$. For example, $\Psi_{\mathrm{3D}}(\sigma_1, \sigma_2, \sigma_3) = \Psi_{\mathrm{3D}}(\sigma_2, \sigma_1, \sigma_3)$.
> All isotropic energies must satisfy this condition.

Our Python code 🐍 above also checks this, and we assure you that it works even $\sigma_i = \sigma_j$ exactly! 🥳
If you’re feeling even lazy 🥱 to write symbolic expressions for $\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma_i}$ and $\frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial \sigma_i \partial \sigma_j}$, you can let a smart automatic differentiator do it for you.
The overhead should be negligible since the variables are only  $\sigma_1$, $\sigma_2$ and $\sigma_3$, and major energies are concisely written in terms of singular values.
This way, we can truly automate 🤖 everything, meaning we won’t need to work on any math 😙 whenever we change the target elastic energy.
Plug and play. 🔌

### 🧭 Using Cauchy-Green Invariants

Picky readers 🤏 point out that singularities still exist $\sigma_i + \sigma_j = 0$, which is a bit extreme since elements need to be inverted to reach this condition, assuming you opt an inversion-aware SVD.
Of course, this occurs frequently in graphics applications, so it should be taken seriously.
The good news 📰 is that we can remove all the singularities in cases where the energies are expressed as a function of Cauchy-Green invariants, as shown by Stomakhin et al. [(2)](#2).
More specifically, this means that the potential is defined as:

```math
\begin{equation}
   \Psi_{\mathrm{3D}} = \Psi(I_1,I_2,I_3). \nonumber
 \end{equation}
 ```

Here, $\Psi_{\mathrm{3D}}$ can be quite complex, even something weird like $\Psi_{\mathrm{3D}} = \frac{(I_1 + I_2 - 6)^2}{I_1} + \sqrt{\frac{I_3}{I_1} + 1} - 2$, is acceptable.
$I_{1 \cdots 3}$ are Cauchy-Green invariants:

```math
\begin{align}
   I_1 &= \mathrm{tr}\left(F^TF\right) = \sigma_1^2 + \sigma_2^2 + \sigma_3^3, \nonumber \\
   I_2 &= \frac{1}{2}\left(\left(I_1\right)^2 - \|F^TF\|^2\right) = (\sigma_1 \sigma_2)^2 + (\sigma_2 \sigma_3)^2 + (\sigma_3 \sigma_1)^2, \nonumber \\
   I_3 &= \mathrm{det}\left(F^TF\right) = (\sigma_1 \sigma_2 \sigma_3)^2. \nonumber
\end{align}
```

With the above definitions, all singularies are cleanly removed such that:

```math
\begin{equation}
  \begin{bmatrix}
      \lambda_1 \\
      \lambda_2 \\
      \lambda_3 \\
      \lambda_4 \\
      \lambda_5 \\
      \lambda_6
  \end{bmatrix} =
  2 \begin{bmatrix}
     1 & \sigma_1 \sigma_2 + \sigma_3^2 & \sigma_1 \sigma_2 \sigma_3^2 \\
     1 & \sigma_1 \sigma_3 + \sigma_2^2 & \sigma_1 \sigma_2^2 \sigma_3 \\
     1 & \sigma_2 \sigma_3 + \sigma_1^2 & \sigma_1^2 \sigma_2 \sigma_3 \\
     1 & \sigma_3^2 - \sigma_1 \sigma_2 & -\sigma_1 \sigma_2 \sigma_3^2 \\
     1 & \sigma_2^2 - \sigma_1 \sigma_3 & -\sigma_1 \sigma_2^2 \sigma_3 \\
     1 & \sigma_1^2 - \sigma_2 \sigma_3 & -\sigma_1^2 \sigma_2 \sigma_3
  \end{bmatrix}
  \begin{bmatrix}
      \frac{\partial}{\partial I_1} \\
      \frac{\partial}{\partial I_2} \\
      \frac{\partial}{\partial I_3}
  \end{bmatrix}
  \Psi_{\mathrm{3D}}. \nonumber
\end{equation}
```

> [!NOTE]
> Of course, if $\frac{\partial \Psi_{\mathrm{3D}}}{\partial I_i}$ includes singularities, we can't do anything about them, but that's not our fault.

And there's more: if you can describe your target energy using invariants, you can **also** express $H_{3 \times 3}$ without directly differentiating $\Psi_{\mathrm{3D}}$ with respect to the singular values:

```math
H_{3 \times 3} = \sum_{i,j} \frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial I_i \partial I_j} \left(\frac{\partial I_i}{\partial \sigma}\right)\left(\frac{\partial I_j}{\partial \sigma}\right)^T + \sum_k \left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial I_k}\right) \left(\frac{\partial^2 I_k}{\partial \sigma^2}\right),
```

where

```math
\frac{\partial I_i}{\partial \sigma} = \begin{bmatrix}
  \displaystyle \frac{\partial}{\partial \sigma_1} \\
  \displaystyle \frac{\partial}{\partial \sigma_2} \\
  \displaystyle \frac{\partial}{\partial \sigma_3}
\end{bmatrix} I_i, \hspace{3mm}
\frac{\partial^2 I_k}{\partial \sigma^2} = \begin{bmatrix}
  \displaystyle \frac{\partial^2}{\partial \sigma_1^2} & \displaystyle \frac{\partial^2}{\partial \sigma_1 \partial \sigma_2} & \displaystyle \frac{\partial^2}{\partial \sigma_1 \partial \sigma_3} \\
 & \displaystyle \frac{\partial^2}{\partial \sigma_2^2} & \displaystyle \frac{\partial^2}{\partial \sigma_2 \partial \sigma_3} \\
 \mathrm{Sym} &  & \displaystyle \frac{\partial^2}{\partial \sigma_3^2}
\end{bmatrix} I_k.
```

To make your life easier 😙, we'll write down explicit expressions:

```math
\begin{align}
\frac{\partial I_1}{\partial \sigma} = 2 \left[\begin{matrix} \sigma_{1}\\ \sigma_{2}\\ \sigma_{3}\end{matrix}\right], \hspace{3mm} \frac{\partial I_2}{\partial \sigma} = 2 \left[\begin{matrix}\sigma_{1} \sigma_{2}^{2} + \sigma_{1} \sigma_{3}^{2}\\\sigma_{1}^{2} \sigma_{2} + \sigma_{2} \sigma_{3}^{2}\\\sigma_{1}^{2} \sigma_{3} + \sigma_{2}^{2} \sigma_{3}\end{matrix}\right], \hspace{3mm} \frac{\partial I_3}{\partial \sigma} = 2 \left[\begin{matrix}\sigma_{1} \sigma_{2}^{2} \sigma_{3}^{2}\\\sigma_{1}^{2} \sigma_{2} \sigma_{3}^{2}\\\sigma_{1}^{2} \sigma_{2}^{2} \sigma_{3}\end{matrix}\right]. \nonumber
\end{align} 
```

```math
\begin{align}
\frac{\partial^2 I_1}{\partial \sigma^2} = 2 \left[\begin{matrix}1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1\end{matrix}\right], \hspace{2mm} \frac{\partial^2 I_2}{\partial \sigma^2} = 2 \left[\begin{matrix}\sigma_{2}^{2} + \sigma_{3}^{2} & 2 \sigma_{1} \sigma_{2} & 2 \sigma_{1} \sigma_{3}\\ & \sigma_{1}^{2} + \sigma_{3}^{2} & 2 \sigma_{2} \sigma_{3}\\ \mathrm{Sym} & & \sigma_{1}^{2} + \sigma_{2}^{2}\end{matrix}\right], \nonumber \\
\frac{\partial^2 I_3}{\partial \sigma^2} = 2 \left[\begin{matrix}\sigma_{2}^{2} \sigma_{3}^{2} & 2 \sigma_{1} \sigma_{2} \sigma_{3}^{2} & 2 \sigma_{1} \sigma_{2}^{2} \sigma_{3}\\ & \sigma_{1}^{2} \sigma_{3}^{2} & 2 \sigma_{1}^{2} \sigma_{2} \sigma_{3}\\ \mathrm{Sym} &  & \sigma_{1}^{2} \sigma_{2}^{2}\end{matrix}\right]. \nonumber
\end{align}
```

The first derivative of $\Psi_{\mathrm{3D}}$ can be **also** computed without $\frac{\partial \Psi_{\mathrm{3D}}}{\partial \sigma}$ using the chain rule:

```math
\frac{\partial \Psi_{\mathrm{3D}}}{\partial f_k} = \sum_k \left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial I_k}\right) \frac{\partial I_k}{\partial f_i},
```

where $C = F^T F$ and

```math
\begin{align}
  \frac{\partial I_1}{\partial F} = 2 F, \hspace{3mm}
  \frac{\partial I_2}{\partial F} = 2 ( \mathrm{tr}\left(C\right) F - F C ), \hspace{3mm}
  \frac{\partial I_3}{\partial F} = 2 F \mathrm{adj} \left(C\right)^T.
\nonumber
\end{align}
```

Just to be clear, $\frac{\partial I_k}{\partial F}$ yields a $3 \times 3$ matrix, and $\frac{\partial I_k}{\partial f_i}$ represents the single entry of $\frac{\partial I_k}{\partial F}$ corresponding to the same row and column as $f_i$.

This way, you will only need to prepare $\frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial I_i \partial I_j}$ and $\frac{\partial \Psi_{\mathrm{3D}}}{\partial I_k}$ whenever you change 🔁 the target energy density.
This approach is often more convenient than working directly with the singular values because many (not all) major energy ⚡ functions are described in terms of Cauchy-Green invariants, and it's also more succinct.
Everything else works without changing your code 🥳.
We wrote a Python script to verify the correctness of the materials in this section [(Code)](../eigsys/eigsys_invariants_3.py).

### 🥼 For Shell Elasticity

We can extend the idea above for triangular meshes.
Recalling that $F$ is $3 \times 2$ for shells, the dimensions of $U$ and $V^T$ change to $U_{3 \times 2}$ and $V^T_{2 \times 2}$, respectively.
If you use our SVD code above, such changes are automatically handled.
How to obrain $F$ from $x$ is mentioned in the paper supplementary.

With this setting, the list of matrices $Q$ and scalars $\lambda$ are given by

```math
\begin{align}
    Q_1 = \frac{1}{\sqrt{2}} U_{3 \times 3}\begin{bmatrix} 0 & 1 \\ -1 & 0 \\ 0 & 0\end{bmatrix} V_{2 \times 2}^T, \hspace{3mm}
    Q_2 = \frac{1}{\sqrt{2}} U_{3 \times 3}\begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0\end{bmatrix} V_{2 \times 2}^T, \nonumber \\
    Q_3 = U_{3 \times 3} \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 1 & 0\end{bmatrix} V_{2 \times 2}^T, \hspace{3mm}
    Q_4 = U_{3 \times 3} \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 1\end{bmatrix} V_{2 \times 2}^T. \nonumber \\
    Q_5 = U_{3 \times 3} \begin{bmatrix}
    a_1 & 0 \\ 0 & a_2 \\ 0 & 0
    \end{bmatrix} V_{2 \times 2}^T, \hspace{3mm}
    Q_6 = U_{3 \times 3} \begin{bmatrix}
    b_1 & 0 \\ 0 & b_2 \\ 0 & 0
    \end{bmatrix} V_{2 \times 2}^T. \nonumber
\end{align}
```

```math
\begin{align}
    \lambda_1 = \frac{1}{\sigma_1+\sigma_2}\left( \frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma_1} + \frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma_2}\right), \nonumber \\
    \lambda_2 = \frac{1}{\sigma_1-\sigma_2}\left( \frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma_1} - \frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma_2}\right), \nonumber \\
    \lambda_3 = \frac{1}{\sigma_1}\left(\frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma_1}\right), \hspace{3mm}
    \lambda_4 = \frac{1}{\sigma_2}\left(\frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma_2}\right). \nonumber
\end{align}
```

Note that our SVD call yields $U_{3 \times 2}$, but the ones used above are $U_{3 \times 3}$. You can convert from $U_{3 \times 2}$ to $U_{3 \times 3}$ by extending it with another column. This additional column can be obtained by taking the cross product of the first and second columns of $U_{3 \times 2}$, such that

```
/* Apache v2.0 License, though this is too simple to claim... */
Matrix3x3 extend_U(const Matrix3x2 &U) {
    Vector3 cross = U.col(0).cross(U.col(1));
    Matrix3x3 result;
    result << U.col(0), U.col(1), cross;
    return result;
}
```

All looks good, except we’re still missing $a_1$, $a_2$, $b_1$, $b_2$, $\lambda_5$, and $\lambda_6$.
Similar to our 3D case, two vectors $[a_1, a_2]$, $[b_1, b_2]$ and their corresponding scalars $\lambda_5$ and $\lambda_6$ are given as the pair of eigenvectors and eigenvalues of the following symmetric $2 \times 2$ matrix:

```math
\begin{equation}
    H_{2 \times 2} = \begin{bmatrix}
    \displaystyle \frac{\partial^2}{\partial \sigma_1^2} & \displaystyle \frac{\partial^2}{\partial \sigma_1 \partial \sigma_2} \\
    \displaystyle \mathrm{Sym} & \displaystyle \frac{\partial^2}{\partial \sigma_2^2}
    \end{bmatrix} \Psi_{\mathrm{2D}}. \nonumber
\end{equation}
```

> [!NOTE]
> When $\Psi_{\mathrm{2D}}$ is written in terms of two Green-Cauchy invariants $I_1 = \mathrm{tr}(F^TF) = \sigma_1^2 + \sigma_2^2$ and $I_2 = \mathrm{det}(F^TF) = \sigma_1^2 \sigma_2^2$ such that:
>
> ```math
> \begin{equation}
>  \Psi_{\mathrm{2D}} = \Psi(I_1,I_2), \nonumber
> \end{equation}
> ```
>
> All singularities can be removed as:
>
> ```math
> \begin{equation}
>   \begin{bmatrix}
>       \lambda_1 \\
>       \lambda_2 \\
>       \lambda_3 \\
>       \lambda_4
>   \end{bmatrix} =
>   2 \begin{bmatrix}
>      1 & \sigma_1 \sigma_2 \\
>      1 & -\sigma_1 \sigma_2 \\
>      1 & \sigma_2^2 \\
>      1 & \sigma_1^2
>   \end{bmatrix}
>   \begin{bmatrix}
>       \frac{\partial}{\partial I_1} \\
>       \frac{\partial}{\partial I_2}
>   \end{bmatrix}
>   \Psi_{\mathrm{2D}}. \nonumber
> \end{equation}
> ```
>
> Similarly to the 3D case, we can leverage the chain rule to obtain $H_{2 \times 2}$ using only the derivatives of $\Psi_{2 \times 2}$ with respect to invariants.
> Derivatives of $I_k$ with respect to singular values are:
>
> ```math
> \begin{align}
> \frac{\partial I_1}{\partial \sigma} = 2 \left[\begin{matrix}\sigma_{1}\\\sigma_{2}\end{matrix}\right],
> \hspace{3mm}
> \frac{\partial I_2}{\partial \sigma} = 2 \left[\begin{matrix}\sigma_{1} \sigma_{2}^{2} + \sigma_{1} \sigma_{3}^{2}\\\sigma_{1}^{2} \sigma_{2} + \sigma_{2} \sigma_{3}^{2}\end{matrix}\right], \nonumber \\
> \hspace{3mm}
> \frac{\partial^2 I_1}{\partial \sigma^2} = 2 \left[\begin{matrix}1 & 0\\0 & 1\end{matrix}\right],
> \hspace{3mm}
> \frac{\partial^2 I_2}{\partial \sigma^2} = 2 \left[\begin{matrix}\sigma_{2}^{2} + \sigma_{3}^{2} & 2 \sigma_{1} \sigma_{2}\\  \mathrm{Sym} & \sigma_{1}^{2} + \sigma_{3}^{2}\end{matrix}\right]. \nonumber
> \end{align}
> ```
>
> The first derivative $\frac{\partial \Psi_{\mathrm{2D}}}{\partial \sigma}$ can be computed similarly to the 3D case. The formula for $\frac{\partial I_k}{\partial F}$ also remains the same.
> Interested readers are referred to our verification Python code [(Code)](../eigsys/eigsys_invariants_2.py).

Everything else is exactly the same as the 3D case presented above.
We provide another Python code [(Code)](../eigsys/eigsys_2.py) to numerically verify this analysis.

## 🎓 Re-Deriving Smith et al. [(1)](#1)

Our technique can be used to arrive at the same eigen system revealed by Smith et al. [(1)](#1).
We can confirm this by simply swapping the Cauchy-Green invariants with those of Smith et al. [(1)](#1) and substitute them into our eigenvalue expressions.
This is simple enough to do manually, but I’ve written a SymPy code to help facilitate the task:

```
from sympy import *

a, b, c = symbols('\sigma_1 \sigma_2 \sigma_3')

I1, I2, I3 = a + b + c, a**2 + b**2 + c**2, a*b*c
E = Function('\Psi')(I1, I2, I3)

display(ratsimp((E.diff(a) + E.diff(b)) / (a + b)))
display(ratsimp((E.diff(c) + E.diff(a)) / (c + a)))
display(ratsimp((E.diff(b) + E.diff(c)) / (b + c)))
display(ratsimp((E.diff(a) - E.diff(b)) / (a - b)))
display(ratsimp((E.diff(c) - E.diff(a)) / (c - a)))
display(ratsimp((E.diff(b) - E.diff(c)) / (b - c)))
```

Now look at Equations (7.16 to 7.21) from [(B)](#B).
The output is identical.
This can't be a coincidence.

<a id="B">[B]</a> Theodore Kim and David Eberle. 2022. Dynamic deformables: implementation and production practicalities (now with code!). In ACM SIGGRAPH 2022 Courses (SIGGRAPH '22). <https://doi.org/10.1145/3532720.3535628>

We have now proven the equivalence of $Q_{1 \cdots 6}$ and $\lambda_{1 \cdots 6}$.
Looks good, but what about scaling mode eigenmatrices $Q_{7 \cdots 9}$ and $\lambda_{7 \cdots 9}$?
Smith et al. [(1)](#1) define $Q_{7,8,9}$ using the eigenvectors and eigenvalues of an encoded matrix $A$ (see their paper for the full representation).
The eigenvectors and eigenvalues are arranged in exactly the same way as ours.
Therefore, to prove that our $Q_{7,8,9}$ and $\lambda_{7,8,9}$ are exactly the same, it is sufficient to show that $H_{3 \times 3} = A$.
For this purpose, let's again use the following chain rule, but this time with $I_k$ being the invariants from Smith et al. [(1)](#1).

```math
H_{3 \times 3} = \sum_{i,j} \frac{\partial^2 \Psi_{\mathrm{3D}}}{\partial I_i \partial I_j} \left(\frac{\partial I_i}{\partial \sigma}\right)\left(\frac{\partial I_j}{\partial \sigma}\right)^T + \sum_k \left(\frac{\partial \Psi_{\mathrm{3D}}}{\partial I_k}\right) \left(\frac{\partial^2 I_k}{\partial \sigma^2}\right),
```

Doing everything by hand is too much work, so let's use SymPy again.
The following code constructs $H_{3 \times 3}$ and $A$ as defined by Smith et al. [(1)](#1) and compares the differences.

```
from sympy import *

a, b, c = symbols('\sigma_1 \sigma_2 \sigma_3')
sym_I1, sym_I2, sym_I3 = symbols('I_1 I_2 I_3')
E = Function('\Psi')(sym_I1, sym_I2, sym_I3)
I1, I2, I3 = a+b+c, a**2+b**2+c**2, a*b*c
I, sym_I, s = [I1,I2,I3], [sym_I1,sym_I2,sym_I3], [a,b,c]

dEdI = Matrix([E.diff(sym_I[i]) for i in range(3)])
d2EdI2 = Matrix([[E.diff(sym_I[i], sym_I[j])
            for i in range(3)]
            for j in range(3)])
dIds = [Matrix([I[k].diff(s[i]) for i in range(3)])
                                for k in range(3)]
d2Ids2 = [Matrix([[I[k].diff(s[i],s[j])
          for i in range(3)] for j in range(3)]) 
          for k in range(3)]

# Our H_3x3 matrix
H3x3 = zeros(3)
for i in range(3):
    H3x3 += dEdI[i] * d2Ids2[i]
    for j in range(3):
        H3x3 += d2EdI2[i,j] * dIds[i] * dIds[j].T

# Smith et al. (2019)
A = zeros(3)
for i in range(3):
    A[i,i] = 2*dEdI[1]+d2EdI2[0,0]+4*s[i]**2*d2EdI2[1,1] \
         + I3**2/s[i]**2*d2EdI2[2,2]+4*s[i]*d2EdI2[0,1] \
         + 4*I3*d2EdI2[1,2]+2*I3/s[i]*d2EdI2[0,2]
for i in range(3):
    for j in range(3):
       if i < j:
           k = 3 - i - j
           A[i,j] = s[k]*dEdI[2]+d2EdI2[0,0] \
             + 4*I3/s[k]*d2EdI2[1,1] \
             + s[k]*I3*d2EdI2[2,2] \
             + 2*s[k]*(I2-s[k]**2)*d2EdI2[1,2] \
             + (I1-s[k])*(s[k]*d2EdI2[0,2]+2*d2EdI2[0,1])
           A[j,i] = A[i,j]

# Symbolically compute the difference
display(simplify(H3x3-A))
```

This prints zero! 😲 But if you tweak the way $A$ is computed a little bit, it gives non-zero expressions 🔢, so this must be correct ✅.
This also confirms that their encoded matrix $A$ corresponds to $\frac{\partial^2 \Psi}{\partial \sigma^2}$.
Now we have proven that our eigen analysis can re-derive the same system as Smith et al. [(1)](#1).

## 🍱 Takeaway C/C++ ⚙️ and Rust 🦀 Codes

If you’re feeling excited, we’d like to share full C/C++ and Rust implementations of the two analyses above under the **Apache v2.0 license**. 📜
Here they are [(📂 Mini Project Directory)](../eigsys).
This also runs on CUDA with minor adaptation work.

