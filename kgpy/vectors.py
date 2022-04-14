"""
Package for easier manipulation of vectors than the usual numpy functions.
"""
import typing as typ
import abc
import dataclasses
import numpy as np
import numpy.typing
import matplotlib.colors
import matplotlib.cm
import matplotlib.lines
import matplotlib.patches
import matplotlib.axes
import mpl_toolkits.mplot3d.art3d
import astropy.units as u
import astropy.visualization
import kgpy.units
import kgpy.labeled
import kgpy.uncertainty
if typ.TYPE_CHECKING:
    import kgpy.matrix

__all__ = [
    'AbstractVector',
    'Cartesian2D',
    'Cartesian3D',
    'Polar',
    'Cylindrical',
    'Spherical',
]

ix = 0
iy = 1
iz = 2

XT = typ.TypeVar('XT', bound=kgpy.uncertainty.ArrayLike)
YT = typ.TypeVar('YT', bound=kgpy.uncertainty.ArrayLike)
ZT = typ.TypeVar('ZT', bound=kgpy.uncertainty.ArrayLike)
CoordinateT = typ.TypeVar('CoordinateT', bound=kgpy.uncertainty.ArrayLike)
ReturnT = typ.TypeVar('ReturnT')
RadiusT = typ.TypeVar('RadiusT', bound=kgpy.uncertainty.ArrayLike)
AzimuthT = typ.TypeVar('AzimuthT', bound=kgpy.uncertainty.ArrayLike)
InclinationT = typ.TypeVar('InclinationT', bound=kgpy.uncertainty.ArrayLike)
AbstractVectorT = typ.TypeVar('AbstractVectorT', bound='AbstractVector')
Cartesian1DT = typ.TypeVar('Cartesian1DT', bound='Cartesian1D')
Cartesian2DT = typ.TypeVar('Cartesian2DT', bound='Cartesian2D')
Cartesian3DT = typ.TypeVar('Cartesian3DT', bound='Cartesian3D')
CartesianNDT = typ.TypeVar('CartesianNDT', bound='CartesianND')
PolarT = typ.TypeVar('PolarT', bound='Polar')
CylindricalT = typ.TypeVar('CylindricalT', bound='Cylindrical')
SphericalT = typ.TypeVar('SphericalT', bound='Spherical')
SpatialSpectralT = typ.TypeVar('SpatialSpectralT', bound='SpatialSpectral')

VectorLike = typ.Union[kgpy.uncertainty.ArrayLike, 'AbstractVector']
ItemArrayT = typ.Union[kgpy.labeled.AbstractArray, kgpy.uncertainty.AbstractArray, AbstractVectorT]


@dataclasses.dataclass(eq=False)
class AbstractVector(
    kgpy.labeled.ArrayInterface,
):
    type_coordinates = kgpy.uncertainty.AbstractArray.type_array + (kgpy.uncertainty.AbstractArray, )

    @classmethod
    def from_coordinates(cls: typ.Type[AbstractVectorT], coordinates: typ.Dict[str, VectorLike]) -> AbstractVectorT:
        return cls(**coordinates)

    @classmethod
    def linear_space(
            cls: typ.Type[AbstractVectorT],
            start: AbstractVectorT,
            stop: AbstractVectorT,
            num: AbstractVectorT,
            endpoint: bool = True,
    ) -> AbstractVectorT:
        coordinates_start = start.coordinates_flat
        coordinates_stop = stop.coordinates_flat
        coordinates_num = num.coordinates_flat
        coordinates_flat = dict()
        for component in coordinates_start:
            coordinates_flat[component] = kgpy.labeled.LinearSpace(
                start=coordinates_start[component],
                stop=coordinates_stop[component],
                num=coordinates_num[component],
                endpoint=endpoint,
                axis=component,
            )

        result = cls()
        result.coordinates_flat = coordinates_flat
        return result

    @classmethod
    def stratified_random_space(
            cls: typ.Type[AbstractVectorT],
            start: AbstractVectorT,
            stop: AbstractVectorT,
            num: AbstractVectorT,
            axis: AbstractVectorT,
            endpoint: bool = True,
            shape_extra: typ.Optional[typ.Dict[str, int]] = None
    ) -> AbstractVectorT:
        coordinates_start = start.coordinates_flat
        coordinates_stop = stop.coordinates_flat
        coordinates_num = num.coordinates_flat
        coordinates_flat = dict()
        for component in coordinates_start:

            if shape_extra is None:
                shape_extra = dict()
            shape = {axis.coordinates_flat[c]: coordinates_num[c] for c in coordinates_num}
            shape_extra_component = {**shape_extra, **shape}
            shape_extra_component.pop(axis.coordinates_flat[component])

            coordinates_flat[component] = kgpy.labeled.StratifiedRandomSpace(
                start=coordinates_start[component],
                stop=coordinates_stop[component],
                num=coordinates_num[component],
                endpoint=endpoint,
                axis=axis.coordinates_flat[component],
                shape_extra=shape_extra_component,
            )

        result = cls()
        result.coordinates_flat = coordinates_flat
        return result

    @property
    def unit(self):
        return getattr(self.coordinates[self.components[0]], 'unit', 1)

    @property
    def coordinates(self: AbstractVectorT) -> typ.Dict[str, VectorLike]:
        return self.__dict__

    @property
    def coordinates_flat(self: AbstractVectorT) -> typ.Dict[str, kgpy.uncertainty.ArrayLike]:
        result = dict()
        coordinates = self.coordinates
        for component in coordinates:
            if isinstance(coordinates[component], AbstractVector):
                coordinates_component = coordinates[component].coordinates_flat
                coordinates_component = {f'{component}.{c}': coordinates_component[c] for c in coordinates_component}
                result = {**result, **coordinates_component}
            else:
                result[component] = coordinates[component]
        return result

    @coordinates_flat.setter
    def coordinates_flat(self: AbstractVectorT, value: typ.Dict[str, kgpy.uncertainty.ArrayLike]):
        coordinates = self.coordinates
        for component in value:
            component_split = component.split('.')
            coordinates_current = coordinates
            for comp in component_split[:~0]:
                coordinates_current = coordinates_current[comp].coordinates

            coordinates_current[component_split[~0]] = value[component]

    @property
    def components(self: AbstractVectorT) -> typ.Tuple[str, ...]:
        return tuple(field.name for field in dataclasses.fields(self))

    @property
    def normalize(self: AbstractVectorT) -> AbstractVectorT:
        other = super().normalized
        for component in other.coordinates:
            if not isinstance(other.coordinates[component], kgpy.labeled.ArrayInterface):
                other.coordinates[component] = kgpy.labeled.Array(other.coordinates[component])
        return other

    @property
    def centers(self: AbstractVectorT) -> AbstractVectorT:
        return self.from_coordinates({c: self.coordinates[c].centers for c in self.coordinates})

    @property
    def array_labeled(self: AbstractVectorT) -> kgpy.labeled.ArrayInterface:
        coordinates = self.broadcasted.coordinates
        return np.stack(list(coordinates.values()), axis='component')

    @property
    def array(self: AbstractVectorT) -> np.ndarray:
        return self.array_labeled.array

    def to(self: AbstractVectorT, unit: u.UnitBase) -> AbstractVectorT:
        other = self.copy_shallow()
        for component in other.coordinates:
            other.coordinates[component] = other.coordinates[component].to(unit)
        return other

    @property
    def tuple(self: AbstractVectorT) -> typ.Tuple[VectorLike, ...]:
        return tuple(self.coordinates.values())

    @property
    def shape(self: AbstractVectorT) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(*self.coordinates.values())

    def astype(
            self: AbstractVectorT,
            dtype: numpy.typing.DTypeLike,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> AbstractVectorT:
        kwargs = dict(
            dtype=dtype,
            order=order,
            casting=casting,
            subok=subok,
            copy=copy,
        )
        coordinates = self.coordinates
        return type(self).from_coordinates({c: coordinates[c].astype(**kwargs) for c in coordinates})

    def __array_ufunc__(self, function, method, *inputs, **kwargs):

        components_result = dict()

        for component in self.components:
            inputs_component = []
            for inp in inputs:
                if isinstance(inp, type(self)):
                    inp = inp.coordinates[component]
                elif isinstance(inp, self.type_coordinates):
                    pass
                else:
                    return NotImplemented
                inputs_component.append(inp)

            for inp in inputs_component:
                if not hasattr(inp, '__array_ufunc__'):
                    inp = np.array(inp)
                try:
                    result = inp.__array_ufunc__(function, method, *inputs_component, **kwargs)
                except ValueError:
                    result = NotImplemented
                if result is not NotImplemented:
                    components_result[component] = result
                    break

            if component not in components_result:
                return NotImplemented

        return type(self).from_coordinates(components_result)

    def __bool__(self: AbstractVectorT) -> bool:
        result = True
        coordinates = self.coordinates
        for component in coordinates:
            result = result and coordinates[component].__bool__()
        return result

    def __mul__(self: AbstractVectorT, other: typ.Union[VectorLike, u.UnitBase]) -> AbstractVectorT:
        if isinstance(other, u.UnitBase):
            coordinates = self.coordinates
            return type(self).from_coordinates({component: coordinates[component] * other for component in coordinates})
        else:
            return super().__mul__(other)

    def __lshift__(self: AbstractVectorT, other: typ.Union[VectorLike, u.UnitBase]) -> AbstractVectorT:
        if isinstance(other, u.UnitBase):
            coordinates = self.coordinates
            return type(self).from_coordinates({component: coordinates[component] << other for component in coordinates})
        else:
            return super().__lshift__(other)

    def __truediv__(self: AbstractVectorT, other: typ.Union[VectorLike, u.UnitBase]) -> AbstractVectorT:
        if isinstance(other, u.UnitBase):
            coordinates = self.coordinates
            return type(self).from_coordinates({component: coordinates[component] / other for component in coordinates})
        else:
            return super().__truediv__(other)

    def __matmul__(self: AbstractVectorT, other: AbstractVectorT) -> AbstractVectorT:
        if isinstance(other, AbstractVector):
            result = 0
            for component in self.coordinates:
                result = result + self.coordinates[component] * other.coordinates[component]
            return result
        else:
            return NotImplemented

    def __array_function__(
            self: AbstractVectorT,
            func: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ) -> AbstractVectorT:

        if func in [
            np.unravel_index,
            np.ndim,
            np.argmin,
            np.nanargmin,
            np.min,
            np.nanmin,
            np.argmax,
            np.nanargmax,
            np.max,
            np.nanmax,
            np.sum,
            np.nansum,
            np.mean,
            np.nanmean,
            np.median,
            np.nanmedian,
            np.percentile,
            np.nanpercentile,
            np.all,
            np.any,
            np.array_equal,
            np.isclose,
            np.roll,
            np.clip,
            np.ptp,
        ]:
            coordinates = dict()
            for component in self.components:
                args_component = [arg.coordinates[component] if isinstance(arg, AbstractVector) else arg for arg in args]
                kwargs_component = {kw: kwargs[kw].coordinates[component] if isinstance(kwargs[kw], AbstractVector) else kwargs[kw] for kw in kwargs}
                coordinates[component] = func(*args_component, **kwargs_component)

            return type(self).from_coordinates(coordinates)

        elif func is np.broadcast_to:
            args = list(args)
            if args:
                array = args[0]
                args.pop(0)
            else:
                array = kwargs['array']
                kwargs.pop('array')
            coordinates = array.coordinates
            coordinates_new = dict()
            for component in coordinates:
                coordinate = coordinates[component]
                if not isinstance(coordinate, kgpy.labeled.ArrayInterface):
                    coordinate = kgpy.labeled.Array(coordinate)
                coordinates_new[component] = np.broadcast_to(coordinate, *args, **kwargs)

            return type(self).from_coordinates(coordinates_new)

        elif func in [np.reshape, np.moveaxis]:
            args = list(args)
            if args:
                array = args.pop(0)
            else:
                array = kwargs.pop('a')

            coordinates = array.coordinates
            coordinates_new = dict()
            for component in coordinates:
                coordinate = coordinates[component]
                if not isinstance(coordinate, kgpy.labeled.ArrayInterface):
                    coordinate = kgpy.labeled.Array(coordinate)
                coordinates_new[component] = func(coordinate, *args, **kwargs)

            return type(self).from_coordinates(coordinates_new)

        elif func in [np.stack, np.concatenate]:
            if args:
                arrays = args[0]
            else:
                arrays = kwargs['arrays']

            coordinates_new = dict()
            for component in arrays[0].coordinates:
                coordinates_new[component] = func([array.coordinates[component] for array in arrays], **kwargs)

            return type(self).from_coordinates(coordinates_new)

        else:
            return NotImplemented

    def __getitem__(
            self: AbstractVectorT,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, ItemArrayT]], ItemArrayT],
    ):
        if isinstance(item, AbstractVector):
            coordinates = {c: self.coordinates[c][item.coordinates[c]] for c in self.coordinates}
        elif isinstance(item, (kgpy.labeled.AbstractArray, kgpy.uncertainty.AbstractArray)):
            coordinates = {c: self.coordinates[c][item] for c in self.coordinates}
        elif isinstance(item, dict):
            if item:
                coordinates = dict()
                for component in self.coordinates:
                    item_component = {k: getattr(item[k], component, item[k]) for k in item}
                    coordinates[component] = self.coordinates[component][item_component]
            else:
                return self
        else:
            raise TypeError
        return type(self).from_coordinates(coordinates)

    def __setitem__(
            self: AbstractVectorT,
            key: typ.Union[typ.Dict[str, typ.Union[int, slice, ItemArrayT]], ItemArrayT],
            value: ItemArrayT,
    ):
        if isinstance(key, type(self)):
            key = key.coordinates
        else:
            key = {component: key for component in self.components}

        if isinstance(value, type(self)):
            value = value.coordinates
        else:
            value = {component: value for component in self.components}

        coordinates = self.coordinates
        for component in coordinates:
            coordinates[component][key[component]] = value[component]

    @property
    def component_sum(self) -> kgpy.uncertainty.ArrayLike:
        result = 0
        coordinates = self.coordinates
        for component in coordinates:
            result = result + coordinates[component]
        return result

    @property
    def length(self) -> kgpy.uncertainty.ArrayLike:
        result = 0
        coordinates = self.coordinates
        for component in coordinates:
            coordinate = coordinates[component]
            if isinstance(coordinate, AbstractVector):
                coordinate = coordinate.length
            result = result + np.square(coordinate)
        result = np.sqrt(result)
        return result

    @property
    def normalized(self: AbstractVectorT) -> AbstractVectorT:
        return self / self.length

    def add_axes(self: AbstractVectorT, axes: typ.List) -> AbstractVectorT:
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component].add_axes(axes=axes)
        return type(self)(**coordinates_new)

    def combine_axes(
            self: AbstractVectorT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> AbstractVectorT:
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component].combine_axes(axes=axes, axis_new=axis_new)
        return type(self)(**coordinates_new)

    def aligned(self: AbstractVectorT, shape: typ.Dict[str, int]):
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component].aligned(shape)
        return type(self)(**coordinates_new)

    def index_nearest_secant(
            self: AbstractVectorT,
            value: AbstractVectorT,
            axis_search: typ.Dict[str, str],
    ) -> typ.Dict[str, kgpy.labeled.Array]:

        import kgpy.optimization

        shape = self.shape
        shape_search = kgpy.vectors.CartesianND({axis: shape[axis] for axis in axis_search})
        indices = self[{ax: 0 for ax in axis_search.values()}].indices

        def indices_factory(index: AbstractVectorT) -> typ.Dict[str, kgpy.labeled.Array]:
            index = np.rint(index).astype(int)
            index = np.clip(index, a_min=0, a_max=shape_search - 1)
            indices = {**indices, **index.coordinates}
            return indices

        def get_index(index: AbstractVectorT) -> AbstractVectorT:
            return self[indices_factory(index)] - value

        result = kgpy.optimization.root_finding.secant(
            func=get_index,
            root_guess=shape_search,
            step_size=kgpy.vectors.CartesianND({ax: 1 for ax in axis_search}),
            max_abs_error=1e-9,
        )

        return indices_factory(result)


    def outer(self: AbstractVectorT, other: AbstractVectorT) -> 'kgpy.matrix.AbstractMatrixT':
        raise NotImplementedError

    def to_matrix(self: AbstractVectorT) -> 'kgpy.matrix.AbstractMatrixT':
        raise NotImplementedError

    def plot(
            self: AbstractVectorT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.lines.Line2D]:
        raise NotImplementedError

    def plot_filled(
            self: AbstractVectorT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.patches.Polygon]:
        raise NotImplementedError


@dataclasses.dataclass(eq=False)
class Cartesian1D(
    AbstractVector,
    typ.Generic[XT],
):
    x: XT = 0

    @classmethod
    def x_hat(cls: typ.Type[Cartesian1DT]) -> Cartesian1DT:
        return cls(x=1)

    def outer(self: Cartesian1DT, other: Cartesian1DT) -> 'kgpy.matrix.AbstractMatrixT':
        raise NotImplementedError

    def to_matrix(self: Cartesian1DT) -> 'kgpy.matrix.AbstractMatrixT':
        import kgpy.matrix
        return kgpy.matrix.Cartesian1D(self.x)

    def plot(
            self: Cartesian1DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.lines.Line2D]:
        raise NotImplementedError

    def plot_filled(
            self: Cartesian1DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.patches.Polygon]:
        raise NotImplementedError


@dataclasses.dataclass(eq=False)
class Cartesian2D(
    Cartesian1D[XT],
    typ.Generic[XT, YT],
):
    y: YT = 0

    @classmethod
    def y_hat(cls: typ.Type[Cartesian2DT]) -> Cartesian2DT:
        return cls(y=1)

    @property
    def polar(self: Cartesian2DT) -> PolarT:
        return Polar(
            radius=np.sqrt(np.square(self.x) + np.square(self.y)),
            azimuth=np.arctan2(self.y, self.x)
        )

    def outer(self: Cartesian2DT, other: Cartesian2DT) -> 'kgpy.matrix.Cartesian2D':
        import kgpy.matrix
        result = kgpy.matrix.Cartesian2D()
        result.x.x = self.x * other.x
        result.x.y = self.x * other.y
        result.y.x = self.y * other.x
        result.y.y = self.y * other.y
        return result

    def to_matrix(self: Cartesian2DT) -> 'kgpy.matrix.Cartesian2D':
        import kgpy.matrix
        return kgpy.matrix.Cartesian2D(
            x=self.x,
            y=self.y,
        )

    def to_3d(self: Cartesian2DT, z: typ.Optional[kgpy.uncertainty.ArrayLike] = None) -> Cartesian3DT:
        if z is None:
            z = 0 * self.x
        return Cartesian3D(
            x=self.x,
            y=self.y,
            z=z,
        )

    @classmethod
    def _broadcast_kwargs(
            cls,
            kwargs: typ.Dict[str, typ.Any],
            shape: typ.Dict[str, int],
            axis_plot: str,
    ):
        shape_kw = shape.copy()
        if axis_plot in shape_kw:
            shape_kw.pop(axis_plot)
        kwargs_broadcasted = dict()
        for k in kwargs:
            kwarg = kwargs[k]
            if not isinstance(kwarg, kgpy.labeled.ArrayInterface):
                kwarg = kgpy.labeled.Array(kwarg)
            kwargs_broadcasted[k] = np.broadcast_to(kwarg, shape_kw)

        return kwargs_broadcasted

    @classmethod
    def _calc_color(
            cls: typ.Type[Cartesian2DT],
            color: typ.Optional[kgpy.labeled.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
    ) -> typ.Tuple[typ.Optional[kgpy.labeled.ArrayLike], typ.Optional[matplotlib.cm.ScalarMappable]]:

        color_array = color.array
        if isinstance(color_array, u.Quantity):
            color_array = color_array.value

        if np.issubdtype(color_array.dtype, np.number):
            if colormap is None:
                colormap = matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(
                        vmin=color_array.min(),
                        vmax=color_array.max(),
                    ),
                    cmap=matplotlib.cm.viridis
                )
            color = kgpy.labeled.Array(np.array(colormap.to_rgba(color_array)), axes=color.axes + ['channel'])

        return color, colormap

    @classmethod
    def _plot_func(
            cls,
            func: typ.Callable[[], typ.List[ReturnT]],
            coordinates: typ.Dict[str, kgpy.labeled.ArrayLike],
            axis_plot: str,
            where: typ.Optional[kgpy.labeled.ArrayLike] = None,
            color: typ.Optional[kgpy.labeled.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            **kwargs: typ.Any,
    ) -> typ.Tuple[typ.List[ReturnT], typ.Optional[matplotlib.cm.ScalarMappable]]:

        with astropy.visualization.quantity_support():

            coordinates = coordinates.copy()
            for component in coordinates:
                if not isinstance(coordinates[component], kgpy.labeled.ArrayInterface):
                    coordinates[component] = kgpy.labeled.Array(coordinates[component])

            if where is None:
                where = True
            if not isinstance(where, kgpy.labeled.ArrayInterface):
                where = kgpy.labeled.Array(where)

            if not isinstance(color, kgpy.labeled.ArrayInterface):
                color = kgpy.labeled.Array(color)

            shape = kgpy.labeled.Array.broadcast_shapes(*coordinates.values(), where, color)
            shape_orthogonal = shape.copy()
            if axis_plot in shape_orthogonal:
                shape_orthogonal.pop(axis_plot)

            for component in coordinates:
                coordinates[component] = coordinates[component].broadcast_to(shape)
            where = where.broadcast_to(shape_orthogonal)
            color = color.broadcast_to(shape_orthogonal)

            color, colormap = cls._calc_color(color, colormap)

            kwargs = cls._broadcast_kwargs(kwargs, shape, axis_plot)

            lines = []
            for index in coordinates[next(iter(coordinates))].ndindex(axis_ignored=axis_plot):
                if where[index]:
                    coordinates_index = {c: coordinates[c][index].array for c in coordinates}
                    kwargs_index = {k: kwargs[k][index].array for k in kwargs}
                    lines += [func(
                        *coordinates_index.values(),
                        color=color[index].array,
                        **kwargs_index
                    )]

        return lines, colormap

    @classmethod
    def _plot_func_uncertainty(
            cls,
            func: typ.Callable[[], typ.List[ReturnT]],
            coordinates: typ.Dict[str, kgpy.uncertainty.ArrayLike],
            axis_plot: str,
            where: typ.Optional[kgpy.uncertainty.ArrayLike] = None,
            color: typ.Optional[kgpy.uncertainty.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            **kwargs: typ.Any,
    ) -> typ.Tuple[typ.List[ReturnT], typ.Optional[matplotlib.cm.ScalarMappable]]:

        uncertain_coordinates = any([isinstance(coordinates[c], kgpy.uncertainty.AbstractArray) for c in coordinates])
        uncertain_where = isinstance(where, kgpy.uncertainty.AbstractArray)
        uncertain_color = isinstance(color, kgpy.uncertainty.AbstractArray)

        if uncertain_coordinates or uncertain_where or uncertain_color:
            coordinates_nominal = dict()
            coordinates_distribution = dict()
            for component in coordinates:
                if isinstance(coordinates[component], kgpy.uncertainty.AbstractArray):
                    coordinates_nominal[component] = coordinates[component].nominal
                    coordinates_distribution[component] = coordinates[component].distribution
                else:
                    coordinates_nominal[component] = coordinates_distribution[component] = coordinates[component]

            if uncertain_where:
                where_nominal = where.nominal
                where_distribution = where.distribution
            else:
                where_nominal = where_distribution = where

            if uncertain_color:
                color_nominal = color.nominal
                color_distribution = color.distribution
            else:
                color_nominal = color_distribution = color

            kwargs_final = dict(
                func=func,
                axis_plot=axis_plot,
                **kwargs
            )
            lines_distribution, colormap = cls._plot_func(
                coordinates=coordinates_distribution,
                where=where_distribution,
                color=color_distribution,
                colormap=colormap,
                **kwargs_final,
            )
            lines_nominal, colormap = cls._plot_func(
                coordinates=coordinates_nominal,
                where=where_nominal,
                color=color_nominal,
                colormap=colormap,
                **kwargs_final,
            )

            lines = lines_nominal + lines_distribution

        else:
            lines, colormap = cls._plot_func(
                func=func,
                coordinates=coordinates,
                where=where,
                axis_plot=axis_plot,
                color=color,
                **kwargs,
            )

        return lines, colormap

    def plot(
            self: Cartesian2DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            where: typ.Optional[kgpy.uncertainty.ArrayLike] = None,
            color: typ.Optional[kgpy.labeled.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            **kwargs: typ.Any,
    ) -> typ.Tuple[typ.List[matplotlib.lines.Line2D], typ.Optional[matplotlib.cm.ScalarMappable]]:

        return self._plot_func_uncertainty(
            func=ax.plot,
            coordinates=self.coordinates,
            where=where,
            axis_plot=axis_plot,
            color=color,
            **kwargs,
        )

    def plot_filled(
            self: Cartesian2DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.patches.Polygon]:

        coordinates = self.coordinates.copy()
        for component in coordinates:
            if isinstance(coordinates[component], kgpy.uncertainty.AbstractArray):
                coordinates[component] = coordinates[component].nominal

        return self._plot_func(
            func=ax.fill,
            coordinates=coordinates,
            axis_plot=axis_plot,
            **kwargs,
        )

    def scatter(
            self: Cartesian2DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            where: typ.Optional[kgpy.uncertainty.ArrayLike] = None,
            color: typ.Optional[kgpy.labeled.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            **kwargs: typ.Any,
    ) -> typ.Tuple[typ.List[matplotlib.lines.Line2D], typ.Optional[matplotlib.cm.ScalarMappable]]:

        return self._plot_func_uncertainty(
            func=ax.scatter,
            coordinates=self.coordinates,
            axis_plot=axis_plot,
            where=where,
            color=color,
            colormap=colormap,
            **kwargs,
        )


@dataclasses.dataclass(eq=False)
class Cartesian3D(
    Cartesian2D[XT, YT],
    typ.Generic[XT, YT, ZT],
):
    z: ZT = 0

    @classmethod
    def z_hat(cls: typ.Type[Cartesian3DT]) -> Cartesian3DT:
        return cls(z=1)

    @property
    def length(self: Cartesian3DT) -> kgpy.uncertainty.ArrayLike:
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    @property
    def xy(self: Cartesian3DT) -> Cartesian2DT:
        return Cartesian2D(
            x=self.x,
            y=self.y,
        )

    @property
    def cylindrical(self: Cartesian3DT) -> CylindricalT:
        return Cylindrical(
            radius=np.sqrt(np.square(self.x) + np.square(self.y)),
            azimuth=np.arctan2(self.y, self.x),
            z=self.z,
        )

    @property
    def spherical(self: Cartesian3DT) -> SphericalT:
        radius = np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))
        return Spherical(
            radius=radius,
            azimuth=np.arctan2(self.y, self.x),
            inclination=np.arccos(self.z / radius)
        )

    def outer(self: Cartesian3DT, other: Cartesian3DT) -> 'kgpy.matrix.Cartesian3D':
        import kgpy.matrix
        result = kgpy.matrix.Cartesian3D()
        result.x.x = self.x * other.x
        result.x.y = self.x * other.y
        result.x.z = self.x * other.z
        result.y.x = self.y * other.x
        result.y.y = self.y * other.y
        result.y.z = self.y * other.z
        result.z.x = self.z * other.x
        result.z.y = self.z * other.y
        result.z.z = self.z * other.z
        return result

    def to_matrix(self: Cartesian3DT) -> 'kgpy.matrix.Cartesian3D':
        import kgpy.matrix
        return kgpy.matrix.Cartesian3D(
            x=self.x,
            y=self.y,
            z=self.z,
        )

    def plot(
            self: Cartesian3DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            where: typ.Optional[kgpy.uncertainty.ArrayLike] = None,
            color: typ.Optional[kgpy.labeled.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: str = 'z',
            **kwargs: typ.Any
    ) -> typ.Tuple[typ.List[matplotlib.lines.Line2D], typ.Optional[matplotlib.cm.ScalarMappable]]:

        if isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
            coordinates = dict(
                x=self.coordinates[component_x],
                y=self.coordinates[component_y],
                z=self.coordinates[component_z],
            )
            return self._plot_func_uncertainty(
                func=ax.plot,
                coordinates=coordinates,
                where=where,
                axis_plot=axis_plot,
                color=color,
                **kwargs,
            )
        else:
            return Cartesian2D(x=self.coordinates[component_x], y=self.coordinates[component_y]).plot(
                ax=ax,
                axis_plot=axis_plot,
                where=where,
                color=color,
                colormap=colormap,
                **kwargs,
            )


    def plot_filled(
        self: Cartesian3DT,
        ax: matplotlib.axes.Axes,
        axis_plot: str,
        component_x: str = 'x',
        component_y: str = 'y',
        component_z: str = 'z',
        **kwargs: typ.Any,
    ) -> typ.List[mpl_toolkits.mplot3d.art3d.Poly3DCollection]:

        coordinates = self.coordinates.copy()
        for component in coordinates:
            if isinstance(coordinates[component], kgpy.uncertainty.AbstractArray):
                coordinates[component] = coordinates[component].nominal

        x, y, z = coordinates[component_x], coordinates[component_y], coordinates[component_z]

        if not isinstance(x, kgpy.labeled.ArrayInterface):
            x = kgpy.labeled.Array(x)
        if not isinstance(y, kgpy.labeled.ArrayInterface):
            y = kgpy.labeled.Array(y)
        if not isinstance(z, kgpy.labeled.ArrayInterface):
            z = kgpy.labeled.Array(z)

        shape = kgpy.labeled.Array.broadcast_shapes(x, y, z)
        x = np.broadcast_to(x, shape)
        y = np.broadcast_to(y, shape)
        z = np.broadcast_to(z, shape)

        kwargs = self._broadcast_kwargs(kwargs, shape, axis_plot)

        if 'color' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('color')
        if 'linewidth' in kwargs:
            kwargs['linewidths'] = kwargs.pop('linewidth')
        if 'linestyle' in kwargs:
            kwargs['linestyles'] = kwargs.pop('linestyle')

        polygons = []
        with astropy.visualization.quantity_support():
            for index in x.ndindex(axis_ignored=axis_plot):
                kwargs_index = {k: kwargs[k][index].array for k in kwargs}
                verts = [np.stack([x[index].array, y[index].array, z[index].array], axis=~0)]
                polygons += [ax.add_collection(mpl_toolkits.mplot3d.art3d.Poly3DCollection(
                    verts=verts,
                    # zsort='max',
                    **kwargs_index,
                ))]

        return polygons


@dataclasses.dataclass(eq=False)
class CartesianND(
    AbstractVector,
    typ.Generic[CoordinateT],
):

    coordinates: typ.Dict[str, CoordinateT] = None

    @classmethod
    def from_coordinates(cls: typ.Type[CartesianNDT], coordinates: typ.Dict[str, CoordinateT]) -> CartesianNDT:
        return cls(coordinates)

    def __post_init__(self: CartesianNDT):
        if self.coordinates is None:
            self.coordinates = dict()

    @property
    def components(self: CartesianNDT) -> typ.Tuple[str, ...]:
        return tuple(self.coordinates.keys())

    def outer(self: CartesianNDT, other: AbstractVectorT) -> 'kgpy.matrix.CartesianND':
        import kgpy.matrix
        coordinates_result = dict()
        coordinates_self = self.coordinates
        coordinates_other = other.coordinates
        for component_self in coordinates_self:
            coordinate_self = coordinates_self[component_self]
            coordinate_result = type(other)(**{c: coordinate_self * coordinates_other[c] for c in coordinates_other})
            coordinates_result[component_self] = coordinate_result
        return kgpy.matrix.CartesianND(coordinates=coordinates_result)

    def to_matrix(self: CartesianNDT) -> 'kgpy.matrix.CartesianND':
        import kgpy.matrix
        return kgpy.matrix.CartesianND(coordinates=self.coordinates)

    def plot(
            self: AbstractVectorT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.lines.Line2D]:
        raise NotImplementedError

    def plot_filled(
            self: AbstractVectorT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.patches.Polygon]:
        raise NotImplementedError


@dataclasses.dataclass(eq=False)
class Polar(
    AbstractVector,
    typ.Generic[RadiusT, AzimuthT],
):
    radius: RadiusT = 0
    azimuth: AzimuthT = 0 * u.deg

    @property
    def length(self: PolarT) -> kgpy.uncertainty.ArrayLike:
        return self.radius

    @property
    def cartesian(self: PolarT) -> Cartesian2DT:
        return Cartesian2D(
            x=self.radius * np.cos(self.azimuth),
            y=self.radius * np.sin(self.azimuth),
        )

    def plot(
            self: PolarT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.lines.Line2D]:
        return self.cartesian.plot(ax=ax, axis_plot=axis_plot, **kwargs)

    def plot_filled(
            self: PolarT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.patches.Polygon]:
        return self.cartesian.plot_filled(ax=ax, axis_plot=axis_plot, **kwargs)


@dataclasses.dataclass(eq=False)
class Cylindrical(
    Polar[RadiusT, AzimuthT],
    typ.Generic[RadiusT, AzimuthT, ZT],
):
    z: ZT = 0

    @property
    def length(self: CylindricalT) -> kgpy.uncertainty.ArrayLike:
        return np.sqrt(np.square(self.radius) + np.square(self.z))

    @property
    def cartesian(self: CylindricalT) -> Cartesian3DT:
        return Cartesian3D(
            x=self.radius * np.cos(self.azimuth),
            y=self.radius * np.sin(self.azimuth),
            z=self.z,
        )
    
    @property
    def spherical(self: CylindricalT) -> SphericalT:
        return Spherical(
            radius=np.sqrt(np.square(self.radius) + np.square(self.z)),
            azimuth=self.azimuth,
            inclination=np.arctan(self.radius / self.z),
        )


@dataclasses.dataclass(eq=False)
class Spherical(
    Polar[RadiusT, AzimuthT],
    typ.Generic[RadiusT, AzimuthT, InclinationT],
):
    inclination: InclinationT = 0 * u.deg

    @property
    def cartesian(self: SphericalT) -> Cartesian3DT:
        return Cartesian3D(
            x=self.radius * np.cos(self.azimuth) * np.sin(self.inclination),
            y=self.radius * np.sin(self.azimuth) * np.sin(self.inclination),
            z=self.radius * np.cos(self.inclination),
        )

    @property
    def cylindrical(self: SphericalT) -> CylindricalT:
        return Cylindrical(
            radius=self.radius * np.sin(self.inclination),
            azimuth=self.azimuth,
            z=self.radius * np.cos(self.inclination),
        )


@dataclasses.dataclass(eq=False)
class SpatialSpectral(
    Cartesian2D,
):
    wavelength: kgpy.uncertainty.ArrayLike = 0 * u.nm

    @property
    def xy(self: SpatialSpectralT) -> Cartesian2D:
        return Cartesian2D(
            x=self.x,
            y=self.y,
        )


@dataclasses.dataclass(eq=False)
class Vector(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):

    @classmethod
    @abc.abstractmethod
    def dimensionless(cls) -> 'Vector':
        return cls()

    @classmethod
    @abc.abstractmethod
    def spatial(cls) -> 'Vector':
        return cls()

    @classmethod
    @abc.abstractmethod
    def angular(cls) -> 'Vector':
        return cls()

    @classmethod
    @abc.abstractmethod
    def from_quantity(cls, value: u.Quantity):
        return cls()

    @classmethod
    def from_tuple(cls, value: typ.Tuple):
        return cls()

    @property
    @abc.abstractmethod
    def quantity(self) -> u.Quantity:
        pass

    @abc.abstractmethod
    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        pass

    @abc.abstractmethod
    def __array_function__(self, function, types, args, kwargs):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def __setitem__(self, key, value):
        pass

    @abc.abstractmethod
    def to_tuple(self):
        pass


@dataclasses.dataclass(eq=False)
class Vector2D(Vector):
    x: numpy.typing.ArrayLike = 0
    y: numpy.typing.ArrayLike = 0

    x_index: typ.ClassVar[int] = 0
    y_index: typ.ClassVar[int] = 1

    __array_priority__ = 100000

    @classmethod
    def dimensionless(cls) -> 'Vector2D':
        self = super().dimensionless()
        self.x = self.x * u.dimensionless_unscaled
        self.y = self.y * u.dimensionless_unscaled
        return self

    @classmethod
    def spatial(cls) -> 'Vector2D':
        self = super().spatial()
        self.x = self.x * u.mm
        self.y = self.y * u.mm
        return self

    @classmethod
    def angular(cls) -> 'Vector2D':
        self = super().angular()
        self.x = self.x * u.deg
        self.y = self.y * u.deg
        return self

    @classmethod
    def from_quantity(cls, value: u.Quantity):
        self = super().from_quantity(value)
        self.x = value[..., cls.x_index]
        self.y = value[..., cls.y_index]
        return self

    @classmethod
    def from_tuple(cls, value: typ.Tuple):
        self = super().from_tuple(value=value)
        self.x = value[ix]
        self.y = value[iy]
        return self

    @classmethod
    def from_cylindrical(
            cls,
            radius: u.Quantity = 0 * u.dimensionless_unscaled,
            azimuth: u.Quantity = 0 * u.deg,
    ) -> 'Vector2D':
        return cls(
            x=radius * np.cos(azimuth),
            y=radius * np.sin(azimuth),
        )

    @property
    def broadcast(self):
        return np.broadcast(self.x, self.y)

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.broadcast.shape

    @property
    def size(self) -> int:
        return self.broadcast.size

    @property
    def ndim(self) -> int:
        return self.broadcast.ndim

    @property
    def x_final(self) -> u.Quantity:
        return np.broadcast_to(self.x, self.shape, subok=True)

    @property
    def y_final(self) -> u.Quantity:
        return np.broadcast_to(self.y, self.shape, subok=True)

    def get_component(self, comp: str) -> u.Quantity:
        return getattr(self, comp)

    def set_component(self, comp: str, value: u.Quantity):
        setattr(self, comp, value)

    @property
    def quantity(self) -> u.Quantity:
        return np.stack([self.x_final, self.y_final], axis=~0)

    @property
    def length_squared(self):
        return np.square(self.x) + np.square(self.y)

    @property
    def length(self):
        return np.sqrt(self.length_squared)

    @property
    def length_l1(self):
        return self.x + self.y

    def normalize(self) -> 'Vector':
        return self / self.length

    @classmethod
    def _extract_attr(cls, values: typ.List, attr: str) -> typ.List:
        values_new = []
        for v in values:
            if isinstance(v, cls):
                values_new.append(getattr(v, attr))
            elif isinstance(v, list):
                values_new.append(cls._extract_attr(v, attr))
            else:
                values_new.append(v)
        return values_new

    @classmethod
    def _extract_attr_dict(cls, values: typ.Dict, attr: str) -> typ.Dict:
        values_new = dict()
        for key in values:
            v = values[key]
            if isinstance(v, cls):
                values_new[key] = getattr(v, attr)
            elif isinstance(v, list):
                values_new[key] = cls._extract_attr(v, attr)
            else:
                values_new[key] = v

        return values_new

    @classmethod
    def _extract_x(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'x')

    @classmethod
    def _extract_y(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'y')

    @classmethod
    def _extract_x_final(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'x_final')

    @classmethod
    def _extract_y_final(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'y_final')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        inputs_x = self._extract_x(inputs)
        for in_x in inputs_x:
            if hasattr(in_x, '__array_ufunc__'):
                result_x = in_x.__array_ufunc__(function, method, *inputs_x, **kwargs)
                if result_x is not NotImplemented:
                    break
        inputs_y = self._extract_y(inputs)
        for in_y in inputs_y:
            if hasattr(in_y, '__array_ufunc__'):
                result_y = in_y.__array_ufunc__(function, method, *inputs_y, **kwargs)
                if result_y is not NotImplemented:
                    break
        if function is np.isfinite:
            return result_x & result_y
        elif function is np.equal:
            return result_x & result_y
        else:
            return type(self)(
                x=result_x,
                y=result_y,
            )

    def __array_function__(self, function, types, args, kwargs):
        if function is np.broadcast_to:
            return self._broadcast_to(*args, **kwargs)
        elif function is np.broadcast_arrays:
            return self._broadcast_arrays(*args, **kwargs)
        elif function is np.result_type:
            return type(self)
        elif function is np.ndim:
            return self.ndim
        elif function in [
            np.min, np.max, np.median, np.mean, np.sum, np.prod,
            np.stack,
            np.moveaxis, np.roll, np.nanmin, np.nanmax,
            np.nansum, np.nanmean, np.linspace, np.where, np.concatenate, np.take
        ]:
            return self._array_function_default(function, types, args, kwargs)
        else:
            raise NotImplementedError

        # args_x = tuple(self._extract_x_final(args))
        # args_y = tuple(self._extract_y_final(args))
        # types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        # types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        # result_x = self.x_final.__array_function__(function, types_x, args_x, kwargs)
        # result_y = self.y_final.__array_function__(function, types_y, args_y, kwargs)
        #
        # if isinstance(result_x, list):
        #     result = [type(self)(x=rx, y=ry) for rx, ry in zip(result_x, result_y)]
        # else:
        #     result = type(self)(x=result_x, y=result_y)
        # return result

    def _array_function_default(self, function, types, args, kwargs):
        args_x = tuple(self._extract_x_final(args))
        args_y = tuple(self._extract_y_final(args))
        kwargs_x = self._extract_attr_dict(kwargs, 'x')
        kwargs_y = self._extract_attr_dict(kwargs, 'y')
        types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        return type(self)(
            x=self.x.__array_function__(function, types_x, args_x, kwargs_x),
            y=self.y.__array_function__(function, types_y, args_y, kwargs_y),
        )

    @classmethod
    def _broadcast_to(cls, value: 'Vector2D', shape: typ.Sequence[int], subok: bool = False) -> 'Vector2D':
        return cls(
            x=np.broadcast_to(value.x, shape, subok=subok),
            y=np.broadcast_to(value.y, shape, subok=subok),
        )

    @classmethod
    def _broadcast_arrays(cls, *args, **kwargs) -> typ.Iterator[numpy.typing.ArrayLike]:
        sh = np.broadcast_shapes(*[a.shape for a in args])
        for a in args:
            yield np.broadcast_to(a, sh, **kwargs)

    @classmethod
    def _min(cls, value: 'Vector2D'):
        return cls(
            x=value.x.min(),

        )

    # def __mul__(self, other) -> 'Vector2D':
    #     return type(self)(
    #         x=self.x.__mul__(other),
    #         y=self.y.__mul__(other),
    #     )
    #
    # def __truediv__(self, other) -> 'Vector2D':
    #     return type(self)(
    #         x=self.x.__truediv__(other),
    #         y=self.y.__truediv__(other),
    #     )

    # def __lshift__(self, other) -> 'Vector2D':
    #     return type(self)(
    #         x=self.x.__lshift__(other),
    #         y=self.y.__lshift__(other),
    #     )

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            return self.x * other.x + self.y * other.y
        else:
            return NotImplementedError

    def __getitem__(self, item):
        return type(self)(
            x=self.x_final.__getitem__(item),
            y=self.y_final.__getitem__(item),
        )

    def __setitem__(self, key, value):
        if isinstance(value, type(self)):
            self.x.__setitem__(key, value.x)
            self.y.__setitem__(key, value.y)
        else:
            self.x.__setitem__(key, value)
            self.y.__setitem__(key, value)

    # def __len__(self):
    #     return self.shape[0]

    def sum(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector2D':
        return np.sum(self, axis=axis, keepdims=keepdims)

    def min(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector2D':
        return np.min(self, axis=axis, keepdims=keepdims)

    def max(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector2D':
        return np.max(self, axis=axis, keepdims=keepdims)

    def reshape(self, *args) -> 'Vector2D':
        return type(self)(
            x=self.x_final.reshape(*args),
            y=self.y_final.reshape(*args),
        )

    def take(self, indices: numpy.typing.ArrayLike, axis: int = None, out: np.ndarray = None, mode: str = 'raise'):
        return np.take(a=self, indices=indices, axis=axis, out=out, mode=mode)

    def outer(self, other: 'Vector2D') -> 'kgpy.matrix.Matrix2D':
        import kgpy.matrix
        result = kgpy.matrix.Matrix2D()
        result.xx = self.x * other.x
        result.xy = self.x * other.y
        result.yx = self.y * other.x
        result.yy = self.y * other.y
        return result

    def to(self, unit: u.Unit) -> 'Vector2D':
        return type(self)(
            x=self.x.to(unit),
            y=self.y.to(unit),
        )

    def to_3d(self, z: typ.Optional[u.Quantity] = None) -> 'Vector3D':
        other = Vector3D()
        other.x = self.x
        other.y = self.y
        if z is None:
            z = 0 * self.x
        other.z = z
        return other

    def to_tuple(self) -> typ.Tuple:
        return self.x, self.y


@dataclasses.dataclass(eq=False)
class Vector3D(Vector2D):
    z: numpy.typing.ArrayLike = 0
    z_index: typ.ClassVar[int] = 2

    __array_priority__ = 1000000

    @classmethod
    def from_quantity(cls, value: u.Quantity):
        self = super().from_quantity(value=value)
        self.z = value[..., cls.z_index]
        return self

    @classmethod
    def from_tuple(cls, value: typ.Tuple):
        self = super().from_tuple(value=value)
        self.z = value[iz]
        return self

    @classmethod
    def dimensionless(cls) -> 'Vector3D':
        self = super().dimensionless()    # type: Vector3D
        self.z = self.z * u.dimensionless_unscaled
        return self

    @classmethod
    def spatial(cls) -> 'Vector3D':
        self = super().spatial()    # type: Vector3D
        self.z = self.z * u.mm
        return self

    @classmethod
    def angular(cls) -> 'Vector3D':
        self = super().angular()    # type: Vector3D
        self.z = self.z * u.deg
        return self

    @classmethod
    def from_cylindrical(
            cls,
            radius: u.Quantity = 0 * u.dimensionless_unscaled,
            azimuth: u.Quantity = 0 * u.deg,
            z: u.Quantity = 0 * u.dimensionless_unscaled
    ) -> 'Vector3D':
        self = super().from_cylindrical(
            radius=radius,
            azimuth=azimuth,
        )  # type: Vector3D
        self.z = z
        return self

    @property
    def xy(self) -> Vector2D:
        return Vector2D(
            x=self.x,
            y=self.y,
        )

    @xy.setter
    def xy(self, value: Vector2D):
        self.x = value.x
        self.y = value.y

    @property
    def yz(self) -> Vector2D:
        return Vector2D(
            x=self.y,
            y=self.z,
        )

    @yz.setter
    def yz(self, value: Vector2D):
        self.y = value.x
        self.z = value.y

    @property
    def zx(self) -> Vector2D:
        return Vector2D(
            x=self.z,
            y=self.x,
        )

    @zx.setter
    def zx(self, value: Vector2D):
        self.z = value.x
        self.x = value.y

    @property
    def broadcast(self):
        return np.broadcast(super().broadcast, self.z)

    @property
    def z_final(self) -> u.Quantity:
        return np.broadcast_to(self.z, self.shape, subok=True)

    @property
    def quantity(self) -> u.Quantity:
        return np.stack([self.x_final, self.y_final, self.z_final], axis=~0)

    @property
    def length_squared(self) -> u.Quantity:
        return super().length_squared + np.square(self.z)

    @property
    def length_l1(self) -> u.Quantity:
        return super().length_l1 + self.z

    @classmethod
    def _extract_z(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'z')

    @classmethod
    def _extract_z_final(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'z_final')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        inputs_z = self._extract_z(inputs)
        for in_z in inputs_z:
            if hasattr(in_z, '__array_ufunc__'):
                result_z = in_z.__array_ufunc__(function, method, *inputs_z, **kwargs)
                if result_z is not NotImplemented:
                    break
        if function is np.isfinite:
            return result & result_z
        elif function is np.equal:
            return result & result_z
        else:
            result.z = result_z
            return result

    # def __array_function__(self, function, types, args, kwargs):
    #     result = super().__array_function__(function, types, args, kwargs)
    #     args_z = tuple(self._extract_z_final(args))
    #     types_z = [type(a) for a in args_z if getattr(a, '__array_function__', None) is not None]
    #     result_z = self.z_final.__array_function__(function, types_z, args_z, kwargs)
    #
    #     if isinstance(result, list):
    #         for r, r_z in zip(result, result_z):
    #             r.z = r_z
    #     else:
    #         result.z = result_z
    #     return result

    def _array_function_default(self, function, types, args, kwargs):
        result = super()._array_function_default(function, types, args, kwargs)
        args_z = tuple(self._extract_z_final(args))
        kwargs_z = self._extract_attr_dict(kwargs, 'z')
        types_z = [type(a) for a in args_z if getattr(a, '__array_function__', None) is not None]
        result.z = self.z.__array_function__(function, types_z, args_z, kwargs_z)
        return result

        # args_x = tuple(self._extract_x_final(args))
        # args_y = tuple(self._extract_y_final(args))
        # types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        # types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        # return type(self)(
        #     x=self.x.__array_function__(function, types_x, args_x, kwargs),
        #     y=self.y.__array_function__(function, types_y, args_y, kwargs),
        # )

    @classmethod
    def _broadcast_to(cls, value: 'Vector3D', shape: typ.Sequence[int], subok: bool = False) -> 'Vector2D':
        result = super()._broadcast_to(value, shape, subok=subok)
        result.z = np.broadcast_to(value.z, shape, subok=subok)
        return result

    # def __mul__(self, other) -> 'Vector3D':
    #     result = super().__mul__(other)     # type: Vector3D
    #     result.z = self.z.__mul__(other)
    #     return result
    #
    # def __truediv__(self, other) -> 'Vector3D':
    #     result = super().__truediv__(other)  # type: Vector3D
    #     result.z = self.z.__truediv__(other)
    #     return result

    # def __lshift__(self, other) -> 'Vector3D':
    #     result = super().__lshift__(other)  # type: Vector3D
    #     result.z = self.z.__lshift__(other)
    #     return result

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            return super().__matmul__(other) + self.z * other.z
        else:
            return NotImplementedError

    def cross(self, other):
        if isinstance(other, type(self)):
            return type(self)(
                x=self.y * other.z - self.z * other.y,
                y=self.z * other.x - self.x * other.z,
                z=self.x * other.y - self.y * other.x,
            )
        else:
            return NotImplementedError

    def __getitem__(self, item):
        other = super().__getitem__(item)
        other.z = self.z_final.__getitem__(item)
        return other

    def __setitem__(self, key, value: 'Vector3D'):
        super().__setitem__(key, value)
        if isinstance(value, type(self)):
            self.z.__setitem__(key, value.z)
        else:
            self.z.__setitem__(key, value)

    def min(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector3D':
        return super().min(axis=axis, keepdims=keepdims)

    def max(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector3D':
        return super().max(axis=axis, keepdims=keepdims)

    def sum(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector3D':
        return super().sum(axis=axis, keepdims=keepdims)

    def reshape(self, *args) -> 'Vector3D':
        result = super().reshape(*args)
        result.z = self.z_final.reshape(*args)
        return result

    def outer(self, other: 'Vector3D') -> 'matrix.Matrix3D':
        result = super().outer(other).to_3d()
        result.xz = self.x * other.z
        result.yz = self.y * other.z
        result.zx = self.z * other.x
        result.zy = self.z * other.y
        result.zz = self.z * other.z
        return result

    def to(self, unit: u.Unit) -> 'Vector3D':
        other = super().to(unit)
        other.z = self.z.to(unit)
        return other

    def to_tuple(self) -> typ.Tuple:
        return super().to_tuple() + self.z


def xhat_factory():
    a = Vector3D.dimensionless()
    a.x = 1 * u.dimensionless_unscaled
    return a


def yhat_factory():
    a = Vector3D.dimensionless()
    a.y = 1 * u.dimensionless_unscaled
    return a


def zhat_factory():
    a = Vector3D.dimensionless()
    a.z = 1 * u.dimensionless_unscaled
    return a


x_hat = xhat_factory()
y_hat = yhat_factory()
z_hat = zhat_factory()
