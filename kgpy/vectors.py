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
import matplotlib.collections
import mpl_toolkits.mplot3d.art3d
import astropy.units as u
import astropy.constants
import astropy.time
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
TimeT = typ.TypeVar('TimeT', bound=kgpy.uncertainty.ArrayLike)

VectorInterfaceT = typ.TypeVar('VectorInterfaceT', bound='VectorInterface')
AbstractVectorT = typ.TypeVar('AbstractVectorT', bound='AbstractVector')
AbstractCartesian1DT = typ.TypeVar('AbstractCartesian1DT', bound='AbstractCartesian1D')
Cartesian1DT = typ.TypeVar('Cartesian1DT', bound='Cartesian1D')
AbstractCartesian2DT = typ.TypeVar('AbstractCartesian2DT', bound='AbstractCartesian2D')
Cartesian2DT = typ.TypeVar('Cartesian2DT', bound='Cartesian2D')
AbstractCartesian3DT = typ.TypeVar('AbstractCartesian3DT', bound='AbstractCartesian3D')
Cartesian3DT = typ.TypeVar('Cartesian3DT', bound='Cartesian3D')
CartesianNDT = typ.TypeVar('CartesianNDT', bound='CartesianND')
PolarT = typ.TypeVar('PolarT', bound='Polar')
CylindricalT = typ.TypeVar('CylindricalT', bound='Cylindrical')
SphericalT = typ.TypeVar('SphericalT', bound='Spherical')

VectorLike = typ.Union[kgpy.uncertainty.ArrayLike, 'VectorInterface']
ItemArrayT = typ.Union[kgpy.labeled.AbstractArray, kgpy.uncertainty.AbstractArray, AbstractVectorT]


@dataclasses.dataclass
class ComponentAxis:
    component: str
    axis: str


@dataclasses.dataclass(eq=False)
class VectorInterface(
    kgpy.labeled.ArrayInterface,
):
    type_coordinates = kgpy.uncertainty.AbstractArray.type_array + (kgpy.uncertainty.AbstractArray,)

    @classmethod
    @abc.abstractmethod
    def prototype(cls: typ.Type[VectorInterfaceT]) -> typ.Type[AbstractVectorT]:
        pass

    @classmethod
    def from_coordinates(cls: typ.Type[VectorInterfaceT], coordinates: typ.Dict[str, VectorLike]) -> AbstractVectorT:
        return cls.prototype()(**coordinates)

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

        result = cls.prototype()()
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

        result = cls.prototype()()
        result.coordinates_flat = coordinates_flat
        return result

    @property
    @abc.abstractmethod
    def coordinates(self: VectorInterfaceT) -> typ.Dict[str, VectorLike]:
        pass

    # @property
    # def coordinates_flat(self: AbstractVectorT) -> typ.Dict[str, kgpy.uncertainty.ArrayLike]:
    #     result = dict()
    #     coordinates = self.coordinates
    #     for component in coordinates:
    #         if isinstance(coordinates[component], AbstractVector):
    #             coordinates_component = coordinates[component].coordinates_flat
    #             coordinates_component = {f'{component}.{c}': coordinates_component[c] for c in coordinates_component}
    #             result = {**result, **coordinates_component}
    #         else:
    #             result[component] = coordinates[component]
    #     return result
    
    @property
    def components(self: VectorInterfaceT) -> typ.List[str]:
        return list(self.coordinates.keys())

    @property
    def array_labeled(self: VectorInterfaceT) -> kgpy.labeled.ArrayInterface:
        coordinates = self.broadcasted.coordinates
        return np.stack(list(coordinates.values()), axis='component')

    @property
    def array(self: VectorInterfaceT) -> np.ndarray:
        return self.array_labeled.array

    @property
    def tuple(self: VectorInterfaceT) -> typ.Tuple[VectorLike, ...]:
        return tuple(self.coordinates.values())

    @property
    def shape(self: VectorInterfaceT) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(*self.coordinates.values())

    @property
    def component_sum(self: VectorInterfaceT) -> kgpy.uncertainty.ArrayLike:
        result = 0
        coordinates = self.coordinates
        for component in coordinates:
            result = result + coordinates[component]
        return result

    @property
    def length(self: VectorInterfaceT) -> kgpy.uncertainty.ArrayLike:
        result = 0
        coordinates = self.coordinates
        for component in coordinates:
            coordinate = coordinates[component]
            if coordinate is None:
                continue
            if isinstance(coordinate, AbstractVector):
                coordinate = coordinate.length
            result = result + np.square(coordinate)
        result = np.sqrt(result)
        return result

    @property
    def normalized(self: VectorInterfaceT) -> VectorInterfaceT:
        return self / self.length

    def astype(
            self: VectorInterfaceT,
            dtype: numpy.typing.DTypeLike,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> VectorInterfaceT:
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
                if inp is None:
                    components_result[component] = None
                    break
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

    def __bool__(self: VectorInterfaceT) -> bool:
        result = True
        coordinates = self.coordinates
        for component in coordinates:
            result = result and coordinates[component].__bool__()
        return result

    def __mul__(self: VectorInterfaceT, other: typ.Union[VectorLike, u.UnitBase]) -> VectorInterfaceT:
        if isinstance(other, u.UnitBase):
            coordinates = self.coordinates
            return type(self).from_coordinates({component: coordinates[component] * other for component in coordinates})
        else:
            return super().__mul__(other)

    def __lshift__(self: VectorInterfaceT, other: typ.Union[VectorLike, u.UnitBase]) -> VectorInterfaceT:
        if isinstance(other, u.UnitBase):
            coordinates = self.coordinates
            return type(self).from_coordinates({component: coordinates[component] << other for component in coordinates})
        else:
            return super().__lshift__(other)

    def __truediv__(self: VectorInterfaceT, other: typ.Union[VectorLike, u.UnitBase]) -> VectorInterfaceT:
        if isinstance(other, u.UnitBase):
            coordinates = self.coordinates
            return type(self).from_coordinates({component: coordinates[component] / other for component in coordinates})
        else:
            return super().__truediv__(other)

    def __matmul__(self: VectorInterfaceT, other: VectorInterfaceT) -> VectorInterfaceT:
        if isinstance(other, VectorInterface):
            if not self.coordinates.keys() == other.coordinates.keys():
                raise ValueError('vectors have different components')
            result = 0
            for component in self.coordinates:
                result = result + self.coordinates[component] * other.coordinates[component]
            return result
        else:
            return NotImplemented

    def __array_function__(
            self: VectorInterfaceT,
            func: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ) -> VectorInterfaceT:

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
            np.std,
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
            self: VectorInterfaceT,
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

    def add_axes(self: VectorInterfaceT, axes: typ.List) -> VectorInterfaceT:
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component].add_axes(axes=axes)
        return type(self).from_coordinates(coordinates_new)

    def combine_axes(
            self: VectorInterfaceT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> VectorInterfaceT:
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component].combine_axes(axes=axes, axis_new=axis_new)
        return type(self).from_coordinates(coordinates_new)

    def aligned(self: VectorInterfaceT, shape: typ.Dict[str, int]) -> AbstractVectorT:
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component].aligned(shape)
        return type(self).from_coordinates(coordinates_new)

    @property
    def type_matrix(self: VectorInterfaceT) -> typ.Type['kgpy.matrix.AbstractMatrix']:
        raise NotImplementedError

    def to_matrix(self: AbstractVectorT) -> 'kgpy.matrix.AbstractMatrixT':
        return self.type_matrix.from_coordinates(self.coordinates.copy())

    def outer(self: AbstractVectorT, other: AbstractVectorT) -> 'kgpy.matrix.AbstractMatrixT':
        coordinates = self.coordinates
        coordinates_new = dict()
        for component in coordinates:
            coordinates_new[component] = coordinates[component] * other
        return self.type_matrix.from_coordinates(coordinates_new)


@dataclasses.dataclass(eq=False)
class AbstractVector(
    VectorInterface,
):

    @classmethod
    def prototype(cls: typ.Type[AbstractVectorT]) -> typ.Type[AbstractVectorT]:
        return cls

    @property
    def unit(self):
        return getattr(self.coordinates[self.components[0]], 'unit', 1)

    @property
    def coordinates(self: AbstractVectorT) -> typ.Dict[str, VectorLike]:
        return self.__dict__

    # @VectorInterface.coordinates_flat.setter
    # def coordinates_flat(self: AbstractVectorT, value: typ.Dict[str, kgpy.uncertainty.ArrayLike]):
    #     coordinates = self.coordinates
    #     for component in value:
    #         component_split = component.split('.')
    #         coordinates_current = coordinates
    #         for comp in component_split[:~0]:
    #             coordinates_current = coordinates_current[comp].coordinates
    #
    #         coordinates_current[component_split[~0]] = value[component]

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

    def to(self: AbstractVectorT, unit: u.UnitBase) -> AbstractVectorT:
        other = self.copy_shallow()
        for component in other.coordinates:
            other.coordinates[component] = other.coordinates[component].to(unit)
        return other

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

    def interp_linear(self: AbstractVectorT, item: typ.Dict[str, VectorLike]):
        coordinates = dict()
        for component in self.coordinates:
            if self.coordinates[component] is not None:
                coordinates[component] = self.coordinates[component].interp_linear(item=item)
            else:
                coordinates[component] = self.coordinates[component]
        return type(self).from_coordinates(coordinates)

    def __call__(self: AbstractVectorT, item: typ.Dict[str, VectorLike]):
        return self.interp_linear(item=item)

    def index_nearest_secant(
            self: AbstractVectorT,
            value: AbstractVectorT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
    ) -> typ.Dict[str, kgpy.labeled.Array]:

        import kgpy.optimization

        if axis is None:
            axis = list(self.shape.keys())
        elif isinstance(axis, str):
            axis = [axis, ]

        shape = self.shape
        shape_nearest = kgpy.vectors.CartesianND({ax: shape[ax] for ax in axis})
        index_base = self.broadcasted[{ax: 0 for ax in axis}].indices

        def indices_factory(index_nearest: AbstractVectorT) -> typ.Dict[str, kgpy.labeled.Array]:
            index_nearest = np.rint(index_nearest).astype(int)
            index_nearest = np.clip(index_nearest, a_min=0, a_max=shape_nearest - 1)
            indices = {**index_base, **index_nearest.coordinates}
            return indices

        def get_index(index: AbstractVectorT) -> AbstractVectorT:
            diff = self.broadcasted[indices_factory(index)] - value
            return kgpy.vectors.CartesianND({c: diff.coordinates[c] for c in diff.coordinates if diff.coordinates[c] is not None})

        result = kgpy.optimization.root_finding.secant(
            func=get_index,
            root_guess=shape_nearest // 2,
            step_size=kgpy.vectors.CartesianND({ax: 1 for ax in axis}),
        )

        return indices_factory(result)

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
class Cartesian(
    VectorInterface,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesian1D(
    Cartesian,
    typ.Generic[XT],
):

    @property
    @abc.abstractmethod
    def x(self: AbstractCartesian1DT) -> kgpy.uncertainty.ArrayLike:
        pass

    @classmethod
    def x_hat(cls: typ.Type[AbstractCartesian1DT]) -> AbstractCartesian1DT:
        return cls.from_coordinates(dict(x=1))

    @property
    def type_matrix(self: AbstractCartesian1DT) -> typ.Type['kgpy.matrix.AbstractMatrix']:
        import kgpy.matrix
        return kgpy.matrix.Cartesian1D


@dataclasses.dataclass(eq=False)
class Cartesian1D(
    AbstractCartesian1D[XT],
    AbstractVector,
):
    x: XT = 0


@dataclasses.dataclass(eq=False)
class AbstractCartesian2D(
    AbstractCartesian1D[XT],
    typ.Generic[XT, YT],
):

    @property
    @abc.abstractmethod
    def y(self: AbstractCartesian2DT) -> kgpy.uncertainty.ArrayLike:
        pass

    @classmethod
    def y_hat(cls: typ.Type[AbstractCartesian2DT]) -> Cartesian1DT:
        return cls.from_coordinates(dict(y=1))

    @property
    def polar(self: AbstractCartesian2DT) -> PolarT:
        return Polar(
            radius=np.sqrt(np.square(self.x) + np.square(self.y)),
            azimuth=np.arctan2(self.y, self.x)
        )

    @property
    def type_matrix(self: AbstractCartesian2DT) -> typ.Type['kgpy.matrix.Cartesian2D']:
        import kgpy.matrix
        return kgpy.matrix.Cartesian2D

    def to_3d(self: AbstractCartesian2DT, z: typ.Optional[kgpy.uncertainty.ArrayLike] = None) -> Cartesian3DT:
        if z is None:
            z = 0 * self.x
        return Cartesian3D(
            x=self.x,
            y=self.y,
            z=z,
        )

    @classmethod
    def _broadcast_kwargs(
            cls: typ.Type[AbstractCartesian2DT],
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
            kwargs_broadcasted[k] = np.broadcast_to(kwarg, {**kwarg.shape, **shape_kw})

        return kwargs_broadcasted

    @classmethod
    def _calc_color(
            cls: typ.Type[AbstractCartesian2DT],
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
            color = kgpy.labeled.Array(
                np.array(colormap.to_rgba(color_array.reshape(-1)).reshape(color_array.shape + (4,))),
                axes=color.axes + ['rgba']
            )

        return color, colormap

    @classmethod
    def _plot_func(
            cls: typ.Type[AbstractCartesian2DT],
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

            shape_kwargs = kgpy.labeled.Array.broadcast_shapes(*kwargs.values(), color)

            shape = kgpy.labeled.Array.broadcast_shapes(*coordinates.values(), where)
            shape_orthogonal = shape.copy()
            if axis_plot in shape_orthogonal:
                shape_orthogonal.pop(axis_plot)

            for component in coordinates:
                coordinates[component] = coordinates[component].broadcast_to(shape)
            where = where.broadcast_to(shape_orthogonal)
            color = color.broadcast_to(shape_kwargs)

            color, colormap = cls._calc_color(color, colormap)

            kwargs = cls._broadcast_kwargs(kwargs, shape_kwargs, axis_plot)

            lines = []
            if shape_orthogonal:
                for index in kgpy.labeled.ndindex(shape_orthogonal):
                    coordinates_index = {c: coordinates[c][index][where[index]].array for c in coordinates}
                    kwargs_index = {k: kwargs[k][index].array for k in kwargs}
                    lines += [func(
                        *coordinates_index.values(),
                        color=color[index].array,
                        **kwargs_index
                    )]
            else:
                coordinates_index = {c: coordinates[c][where].array for c in coordinates}
                lines += [func(
                    *coordinates_index.values(),
                    color=color[dict()].array,
                )]

        return lines, colormap

    @classmethod
    def _plot_func_uncertainty(
            cls: typ.Type[AbstractCartesian2DT],
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
            self: AbstractCartesian2DT,
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
            self: AbstractCartesian2DT,
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
            self: AbstractCartesian2DT,
            ax: matplotlib.axes.Axes,
            axis_plot: str,
            where: typ.Optional[kgpy.uncertainty.ArrayLike] = None,
            color: typ.Optional[kgpy.labeled.ArrayLike] = None,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            **kwargs: typ.Any,
    ) -> typ.Tuple[typ.List[matplotlib.collections.PathCollection], typ.Optional[matplotlib.cm.ScalarMappable]]:

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
class Cartesian2D(
    AbstractCartesian2D[XT, YT],
    Cartesian1D[XT],
):
    y: YT = 0


@dataclasses.dataclass(eq=False)
class AbstractCartesian3D(
    AbstractCartesian2D[XT, YT],
    typ.Generic[XT, YT, ZT],
):

    @property
    @abc.abstractmethod
    def z(self: AbstractCartesian3DT) -> kgpy.uncertainty.ArrayLike:
        pass

    @classmethod
    def z_hat(cls: typ.Type[AbstractCartesian3DT]) -> Cartesian3DT:
        return cls.from_coordinates(dict(z=1))

    @property
    def xy(self: AbstractCartesian3DT) -> Cartesian2DT:
        return Cartesian2D(
            x=self.x,
            y=self.y,
        )

    @property
    def zx(self) -> Cartesian2DT:
        return Cartesian2D(
            x=self.z,
            y=self.x,
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

    @property
    def type_matrix(self: AbstractCartesian3DT) -> typ.Type['kgpy.matrix.Cartesian3D']:
        import kgpy.matrix
        return kgpy.matrix.Cartesian3D

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
class Cartesian3D(
    AbstractCartesian3D[XT, YT, ZT],
    Cartesian2D[XT, YT],
    typ.Generic[XT, YT, ZT],
):
    z: ZT = 0


@dataclasses.dataclass(eq=False)
class CartesianND(
    Cartesian,
    AbstractVector,
    typ.Generic[CoordinateT],
):

    coordinates: typ.Dict[str, CoordinateT] = None

    # def __getattr__(self: CartesianNDT, item: str):
    #     if item in self.coordinates:
    #         return self.coordinates[item]
    #     if item == 'coordinates':
    #         return self.coordinates
    #     else:
    #         return self.coordinates[item]

    @classmethod
    def from_coordinates(cls: typ.Type[CartesianNDT], coordinates: typ.Dict[str, CoordinateT]) -> CartesianNDT:
        return cls(coordinates)

    @classmethod
    def from_components(cls: typ.Type[CartesianNDT], **components: CoordinateT) -> CartesianNDT:
        return cls.from_coordinates(components)

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

    def copy_shallow(self: CartesianNDT) -> CartesianNDT:
        return type(self)(self.coordinates.copy())

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
class TemporalVector(
    AbstractVector,
    typ.Generic[TimeT],
):
    time: TimeT = astropy.time.Time(0, format='unix')



