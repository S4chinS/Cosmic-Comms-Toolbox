## How to compute $C_{d}A$ and $C_{L}A$ from CLL / Sentman (diffuse)

### Goal
Compute **$C_{d}A$** and **$C_{L}A$** (effective areas, units m²) from either a **CLL** gas–surface interaction (GSI) model or a **Sentman diffuse** GSI model for an arbitrary spacecraft attitude and flow vector, suitable for use in an orbit propagator.

This write-up is intentionally **codebase-agnostic**: it describes the general process, frames, and projections you need to implement.

---

## 1) Frames and required inputs

At each time step you need:

- **Geometry in a body frame**: a triangular mesh. For each triangle/face $f$:
  - area $A_f$
  - unit normal in **body frame** $\hat n_{b,f}$
  - centroid $r_{b,f}$ (optional, for torque)

- **Attitude**: rotation matrix $R_{wb}$ mapping **body → world**:
  - $\hat n_{w,f} = R_{wb}\hat n_{b,f}$

- **Relative flow velocity in world frame**: $\mathbf{v}_{rel,w}$
  - unit direction $\hat v_w = \mathbf{v}_{rel,w}/\|\mathbf{v}_{rel,w}\|$
  - speed $V=\|\mathbf{v}_{rel,w}\|$

Environmental inputs (typical free-molecular / GSI setup):

- Density $\rho$
- Gas temperature $T$
- Species number densities $n_i$ and particle masses $m_i$
- Wall temperature $T_w$
- For CLL: accommodation coefficients (normal and tangential) per species (or equivalent parameters)

---

## 2) Per-face incidence angle and tangential direction

For each face $f$:

### 2.1 Transform normal to world
$$
\hat n_{w,f} = R_{wb}\,\hat n_{b,f}
$$

### 2.2 Compute incidence angle
Define incidence as the angle between the face normal and the **incoming** flow direction $-\hat v_w$:
$$
\cos\theta_f = \hat n_{w,f}\cdot(-\hat v_w)
$$
Clamp $\cos\theta_f\in[-1,1]$, then:
$$
\theta_f = \arccos(\cos\theta_f)
$$

### 2.3 Compute tangential direction in the face plane (flow-aligned)
Define a unit tangent direction aligned with the in-plane component of the flow:
$$
\hat t_{w,f} \propto \big((\hat v_w \times \hat n_{w,f}) \times \hat n_{w,f}\big)
$$
Normalize if nonzero. If the flow is exactly normal to the face, the tangential direction is undefined; you can set tangential contribution to zero for that face (it won’t matter).

---

## 3) Call the GSI model to get $(C_n(\theta), C_t(\theta))$

For each face, call your chosen model with $\theta_f$ and environment:

### CLL (accommodation model)
Returns:
- $C_{n,f}$: normal coefficient (pressure-like)
- $C_{t,f}$: tangential coefficient (shear-like)

These are computed from modelled normal/tangential momentum fluxes, then normalized by the dynamic pressure:
$$
q_\infty = \tfrac12 \rho V^2
$$

### Sentman diffuse (fully diffuse / fully accommodating)
Also returns:
- $C_{n,f}$
- $C_{t,f}$

Here, normal comes from incident + diffuse-reflected normal momentum flux; tangential is from incident tangential momentum flux with **zero reflected shear** in the diffuse limit.

Key point: **both models yield coefficients defined relative to the local face frame** (normal / tangential), not global “drag/lift” axes.

---

## 4) Convert $(C_n, C_t)$ to per-face effective area vectors ($A^\*$)

Multiply by face area $A_f$ to get effective areas:

- $C_{n,f}A_f$ and $C_{t,f}A_f$ have units **m²**

Build a **world-frame effective area vector** contribution:
$$
\mathbf{A}^\*_{w,f} = s\Big[(C_{n,f}A_f)\,\hat n_{w,f} \;+\; (C_{t,f}A_f)\,\hat t_{w,f}\Big]
$$
where $s\in\{+1,-1\}$ is a **sign convention**. Pick $s$ so that the resulting total drag (defined below) is positive. Many implementations choose $s=-1$ so the vector opposes the flow components.

---

## 5) Sum over faces to get total effective area vector (world frame)

$$
\mathbf{A}^\*_{w} = \sum_f \mathbf{A}^\*_{w,f}
$$

This is a **3D vector in the world/global frame** with units **m²**.

This is often the best object to pass around: it cleanly separates **geometry+attitude** (in $\mathbf{A}^\*$) from the environmental scaling $q_\infty$.

---

## 6) Project the effective area vector to get $C_{d}A$ and $C_{L}A$

With unit flow direction $\hat v_w$:

### Effective drag area
Define:
$$
C_{d}A = \mathbf{A}^\*_w \cdot \hat v_w
$$
If this comes out negative, flip the sign convention $s$ (or equivalently use $-\hat v_w$) so that **$C_{d}A\ge 0$** for your standard cases.

### Effective lift-like area (perpendicular magnitude)
Compute the component perpendicular to the flow:
$$
\mathbf{A}^\*_\perp = \mathbf{A}^\*_w - (C_{d}A)\,\hat v_w
$$
Then:
$$
C_{L}A = \|\mathbf{A}^\*_\perp\|
$$

Notes:
- This $C_{L}A$ is a *magnitude* of everything perpendicular to the flow. If you need signed lift in a particular plane, define a lift axis $\hat \ell_w$ (e.g., orbit-normal or a vehicle “up” direction in world frame) and compute $C_{L,\text{signed}}A = \mathbf{A}^\*_w\cdot\hat \ell_w$.
- If $\hat v_w$ is aligned with a principal axis (e.g., $\hat v_w=[-1,0,0]$), then the dot-product reduces to a component pick (up to sign), but using the projection keeps the method robust for general orientations.

---

## 7) Convert to force / acceleration for orbit propagation

Dynamic pressure:
$$
q_\infty = \tfrac12 \rho \|\mathbf{v}_{rel}\|^2
$$

Force vector:
$$
\mathbf{F}_w = q_\infty\,\mathbf{A}^\*_w
$$

Drag magnitude:
$$
F_D = q_\infty\,C_{d}A
$$

Lift-like magnitude:
$$
F_L = q_\infty\,C_{L}A
$$

Acceleration:
$$
\mathbf{a}_w = \mathbf{F}_w / m_{sc}
$$

---

## 8) Practical integration notes

- Keep $\mathbf{v}_{rel}$, $R_{wb}$, and $\mathbf{A}^\*_w$ in the **same frame** (ECI/ECEF/LVLH—any is fine if consistent).
- Do not assume “drag is X and lift is Z” unless you explicitly align axes and define lift accordingly.
- A good propagator interface is returning **$\mathbf{A}^\*_w$**; $C_{d}A$ and $C_{L}A$ are derived projections.
- Optional torque:
  - Per-face force: $\mathbf{F}_{w,f}=q_\infty\mathbf{A}^\*_{w,f}$
  - Sum: $\boldsymbol{\tau}_w=\sum_f (r_{w,f}-r_{w,CoM})\times \mathbf{F}_{w,f}$

---

### Summary (one-line algorithm)
Transform face normals to world → compute $\theta$ and $\hat t$ from flow direction → call CLL/Sentman to get $(C_n,C_t)$ → form per-face $\mathbf{A}^\*$ vectors → sum to $\mathbf{A}^\*_w$ → project onto $\hat v_w$ to get $C_{d}A$ and onto perpendicular space to get $C_{L}A$.

