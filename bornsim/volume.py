"""Finite refractive-index fields on centred cubic voxel grids."""

from dataclasses import dataclass
import numpy as np
from ._validation import _integer
from .grid import Grid
from .media import RandomMedium
from .units import _refractive_index_values, Quantity


@dataclass(frozen=True, kw_only=True)
class Volume:
    """Represent a finite refractive-index fluctuation field on cubic voxels.

    Parameters
    ----------
    delta_refractive_index : array_like
        Finite three-dimensional refractive index fluctuations relative to the background,
        with 2 to 32 cells per axis. Quantities are rejected, including dimensionless ones.
    grid : Grid, optional
        Shared spatial configuration. Its shape must match delta_refractive_index.
        Supply either grid or spacing.
    spacing : Quantity, optional
        Positive, finite cubic voxel width. Explicit length units are required.
        Legacy alternative to supplying grid; shape is inferred from the array.
    background_refractive_index : float
        Positive, finite background refractive index. Plain numbers are required; quantities are rejected. A background must be supplied.
    medium : RandomMedium, optional
        Generation statistics recorded by ``random_volume``. None for a
        manually supplied field. Must be supplied together with ``seed``.
    seed : int, optional
        Random seed recorded by ``random_volume``. None for a manual field.

    Attributes
    ----------
    delta_refractive_index : numpy.ndarray
        Copied, read-only, dimensionless fluctuation array of shape (nx, ny, nz).
    grid : Grid
        Spatial configuration shared with generation and scattering.
    spacing : Quantity
        Cubic voxel width with its supplied units, also available as grid.spacing.
    background_refractive_index : float
        Dimensionless background refractive index.
    medium : RandomMedium or None
        Recorded generation statistics, when available.
    seed : int or None
        Recorded generation seed, when available.

    Raises
    ------
    ValueError
        If the field or grid is invalid, units are incompatible, or linearized
        relative permittivity is nonpositive in any voxel.
    TypeError
        If recorded generation statistics are not a RandomMedium.

    See Also
    --------
    bornsim.media.random_volume : Generate a seeded sample from medium statistics.
    bornsim.series.BornSeries.solve : Compute coherent scattering from the sample.

    Notes
    -----
    Relative permittivity is ``background_refractive_index**2 + 2*background_refractive_index*delta_refractive_index``.
    The quadratic refractive-index fluctuation term is omitted at every Born order.
    Voxel centres are measured relative to the sample centre.

    Examples
    --------
    >>> import numpy as np
    >>> from bornsim import Volume
    >>> from bornsim.units import ureg
    ...
    ...
    >>> volume = Volume(
    ...     delta_refractive_index=np.zeros((2, 2, 2)),
    ...     spacing=50 * ureg.nanometer,
    ...     background_refractive_index=1.33,
    ... )
    >>> volume.positions.shape
    (2, 2, 2, 3)
    """

    delta_refractive_index: np.ndarray
    background_refractive_index: float
    spacing: Quantity = None
    grid: Grid | None = None
    medium: RandomMedium | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        data = np.array(
            _refractive_index_values(
                value=self.delta_refractive_index,
                name="delta_refractive_index",
            ),
            copy=True,
        )

        if data.ndim != 3 or any(n < 2 or n > 32 for n in data.shape) or not np.all(np.isfinite(data)):
            raise ValueError("delta_refractive_index must be a finite 3D array with 2–32 cells per axis.")

        if self.grid is None:
            if self.spacing is None:
                raise ValueError("Supply grid or spacing for a Volume.")

            grid = Grid(
                shape=data.shape,
                spacing=self.spacing,
            )
        else:
            grid = Grid._resolve(
                grid=self.grid,
                spacing=self.spacing,
            )

            if grid.shape != data.shape:
                raise ValueError("grid shape must match delta_refractive_index.")

        object.__setattr__(self, "grid", grid)

        object.__setattr__(self, "spacing", grid.spacing)

        object.__setattr__(
            self,
            "background_refractive_index",
            _refractive_index_values(
                value=self.background_refractive_index,
                name="background_refractive_index",
                scalar=True,
            ),
        )

        if not np.isfinite(self.background_refractive_index) or self.background_refractive_index <= 0:
            raise ValueError("background_refractive_index must be finite and positive.")

        if np.any(self.background_refractive_index**2 + 2 * self.background_refractive_index * data <= 0):
            raise ValueError("Linearized relative permittivity must stay positive.")

        if (self.medium is None) != (self.seed is None):
            raise ValueError("medium and seed must be supplied together.")

        if self.medium is not None:
            if not isinstance(self.medium, RandomMedium):
                raise TypeError("medium must be a RandomMedium.")

            if self.medium.background_refractive_index != self.background_refractive_index:
                raise ValueError("medium background_refractive_index must match the volume.")

            object.__setattr__(
                self,
                "seed",
                _integer(
                    value=self.seed,
                    name="seed",
                    low=0,
                    high=2**32 - 1,
                ),
            )

        data.setflags(write=False)

        object.__setattr__(self, "delta_refractive_index", data)

    @property
    def positions(self) -> Quantity:
        """Return voxel-centre coordinates relative to the sample centre.

        Returns
        -------
        positions : Quantity
            Unit-bearing coordinates, shape (nx, ny, nz, 3), with the final
            axis ordered as x, y, z.
        """

        return getattr(self, "grid").positions

    @property
    def volume(self) -> Quantity:
        """Return the total physical volume of all voxels.

        Returns
        -------
        volume : Quantity
            Unit-bearing sample volume, equal to ``delta_refractive_index.size*spacing**3``.
        """

        return getattr(self, "grid").volume

    def plot_slice(self, *, normal="z", index=None, field="refractive_index", length_unit="nanometer", title=None):
        """Build a Matplotlib figure of one voxel plane.

        Parameters
        ----------
        normal : {'x', 'y', 'z'}, optional
            Axis perpendicular to the plane; default 'z'.
        index : int, optional
            Voxel index along the normal. Defaults to n//2, the positive
            central plane for an even grid. The title gives its actual position.
        field : {'refractive_index', 'delta_refractive_index', 'permittivity'}, optional
            Display n0 + delta_refractive_index, fluctuations, or linearized relative
            permittivity n0**2 + 2*n0*delta_refractive_index. Default 'refractive_index'.
        length_unit : str, optional
            Spatial display unit; default 'nanometer'. Stored SI data is unchanged.
        title : str, optional
            Override the title describing the field and plane position.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Equal-aspect image with voxel-edge extents and a labelled colorbar.
            Call pyplot.show() to display or figure.savefig() to export.
            Colors use the full volume's range to compare different planes;
            fluctuation colors are symmetric about zero.

        Raises
        ------
        ValueError
            If the plane, index, field, or length unit is invalid.
        """

        from ._volume_plotting import _VolumePlotter

        return _VolumePlotter(
            volume=self,
        ).plot_slice(
            normal=normal,
            index=index,
            field=field,
            length_unit=length_unit,
            title=title,
        )

    def plot_3d(
        self,
        *,
        backend="plotly",
        mode=None,
        field="delta_refractive_index",
        length_unit="nanometer",
        surface_count=8,
        opacity=None,
        opacity_scale="uniform",
        slice_indices=None,
    ):
        """Build a three-dimensional figure of the finite voxel medium.

        Parameters
        ----------
        backend : {'matplotlib', 'plotly'}, optional
            Default is Plotly for browser interaction. Select 'matplotlib'
            explicitly for a Matplotlib figure.
        mode : str, optional
            Matplotlib supports 'slices' (default) and 'voxels'. Voxels display
            cells with nonzero refractive index contrast, or the full box for a zero field.
            Plotly supports 'volume' (default), 'isosurface', and 'slices'.
            Uniform Plotly fields fall back to slices.
        field : {'delta_refractive_index', 'refractive_index', 'permittivity'}, optional
            Scalar field to display, default refractive index fluctuation. 'refractive_index' displays
            n0 + delta_refractive_index; 'permittivity' displays the solver's linearized
            relative permittivity n0**2 + 2*n0*delta_refractive_index, not (n0+delta_refractive_index)**2.
        length_unit : str, optional
            Spatial display unit, default 'nanometer'. Stored SI data is unchanged.
        surface_count : int, optional
            Number of contours, from 1 to 32; default 8. More surfaces increase
            browser rendering cost. Used only by Plotly volume and isosurface modes.
        opacity : float, optional
            Surface opacity in (0, 1]. Default is 1 for Matplotlib and 0.15
            for Plotly. Slices are opaque in both backends.
        opacity_scale : {'uniform', 'increasing'}, optional
            Default 'uniform' gives all contours the same opacity. 'increasing'
            is available only for Plotly volume mode and scales opacity from
            zero at the lowest displayed contour to ``opacity`` at the highest.
            Use field='refractive_index' to emphasize high refractive index, rather than
            the magnitude of positive and negative fluctuations.
        slice_indices : tuple of int, optional
            One voxel index for each x, y, z plane, used only for slices.
            Defaults to the middle voxel along each axis.

        Returns
        -------
        figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Matplotlib: call pyplot.show() to display or figure.savefig() to
            export. Rotation needs an interactive Matplotlib backend; gallery
            images are static. Plotly: call figure.show() or figure.write_html().
            Constructing a figure opens no window or browser.

        Raises
        ------
        ValueError
            If a mode, scalar field, display unit, or rendering setting is invalid.

        Notes
        -----
        Matplotlib voxels preserve staircase interfaces; slices display every
        selected cell with voxel-edge extents. Dense transparent scenes can have
        depth-ordering limitations in Matplotlib. Plotly contours interpolate
        between voxel centres and are visualization aids,
        not exact curved interfaces or a replacement for grid refinement.
        Slices show sampled values. Aspect ratio follows physical dimensions.
        All voxels are retained, with at most 32**3 samples under Volume's grid
        limit; browser performance also depends on surface count and graphics
        support. This plots the input material, not an electromagnetic field.
        """

        from ._volume_plotting import _VolumePlotter

        return _VolumePlotter(
            volume=self,
        ).plot_3d(
            backend=backend,
            mode=mode,
            field=field,
            length_unit=length_unit,
            surface_count=surface_count,
            opacity=opacity,
            opacity_scale=opacity_scale,
            slice_indices=slice_indices,
        )
