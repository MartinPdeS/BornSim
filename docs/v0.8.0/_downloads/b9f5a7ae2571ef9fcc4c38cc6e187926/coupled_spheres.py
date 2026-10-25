"""
Higher-order coupling in a pair of spheres
==========================================

Compare a jointly solved sphere pair with the coherent sum of isolated
sphere amplitudes. At first order these coincide. Higher orders in the joint
solve also include propagation between particles; isolated solves omit it.
The spheres do not overlap. This remains a linearized dielectric model.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg

grid = Grid(
    shape=(12, 12, 12),
    spacing=20 * ureg.nanometer,
)

left_sphere = Sphere(
    radius=40 * ureg.nanometer,
    refractive_index=1.37,
    centre=[-60, 0, 0] * ureg.nanometer,
)

right_sphere = Sphere(
    radius=40 * ureg.nanometer,
    refractive_index=1.37,
    centre=[60, 0, 0] * ureg.nanometer,
)

source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(
    source=source,
    order=3,
)

angles = np.linspace(0, 180, 91) * ureg.degree

results = []

for structures in ((left_sphere,), (right_sphere,), (left_sphere, right_sphere)):
    medium = StructuredMedium()

    medium.add_background(refractive_index=1.33)

    medium.add_structures(*structures)

    volume = medium.to_volume(grid=grid)

    volume.plot_slice()

    plt.show()

    result = solver.solve_cut(
        target=volume,
        angles=angles,
    )

    results.append(result)

# %%
# Compare amplitudes before squaring; average incident polarizations afterward.
independent_amplitudes = results[0].amplitudes + results[1].amplitudes

independent_intensity = np.mean(np.sum(abs(independent_amplitudes.to("nanometer").magnitude) ** 2, axis=-1), axis=-1)

joint_intensity = results[2].differential_cross_section.to("nanometer**2 / steradian").magnitude

figure, axis = plt.subplots(layout="constrained")

for order in (1, 3):
    axis.plot(angles.magnitude, joint_intensity[order - 1], label=f"Joint, order {order}")

    axis.plot(angles.magnitude, independent_intensity[order - 1], "--", label=f"Isolated amplitude sum, order {order}")

axis.set(xlabel="Scattering angle (degrees)", ylabel="Cross section (nm² sr⁻¹)")

axis.legend()

plt.show()
