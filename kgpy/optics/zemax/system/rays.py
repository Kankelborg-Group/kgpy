
import enum
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization

from kgpy.optics.zemax import ZOSAPI

RayType = tp.List[tp.List[tp.List[tp.List[tp.List[tp.List[tp.List[tp.List[float]]]]]]]]
Hist1DType = tp.Tuple[np.ndarray, np.ndarray]
Hist2DType = tp.Tuple[np.ndarray, np.ndarray, np.ndarray]

class RayAxes(enum.IntEnum):

    grid = enum.auto()
    field = enum.auto()
    pupil = enum.auto()


class GridIndices(enum.IntEnum):

    field_x = enum.auto()
    field_y = enum.auto()
    pupil_x = enum.auto()
    pupil_y = enum.auto()


class ReturnIndices(enum.IntEnum):

    error = enum.auto()
    vignetting = enum.auto()
    x = enum.auto()
    y = enum.auto()


class RayIndices(enum.IntEnum):

    intensity = enum.auto()
    fx = enum.auto()
    fy = enum.auto()
    px = enum.auto()
    py = enum.auto()
    x = enum.auto()
    y = enum.auto()


def trace_to_image(zemax_system: ZOSAPI.IOpticalSystem,
                   field_positions_per_axis: tp.Union[int, tp.Tuple[int, int]],
                   pupil_positions_per_axis: tp.Union[int, tp.Tuple[int, int]],
                   ) -> RayType:

    if isinstance(field_positions_per_axis, int):
        field_positions_per_axis = (field_positions_per_axis, field_positions_per_axis)

    if isinstance(pupil_positions_per_axis, int):
        pupil_positions_per_axis = (pupil_positions_per_axis, pupil_positions_per_axis)

    field_list_x = np.linspace(-1.0, 1.0, field_positions_per_axis[0])
    field_list_y = np.linspace(-1.0, 1.0, field_positions_per_axis[1])

    pupil_list_x = np.linspace(-1.0, 1.0, pupil_positions_per_axis[0])
    pupil_list_y = np.linspace(-1.0, 1.0, pupil_positions_per_axis[1])

    intensity = 1.0

    rays = []

    for c in range(zemax_system.MCE.NumberOfConfigurations):

        rays_c = []

        for s in [zemax_system.LDE.NumberOfSurfaces - 1]:

            rays_s = []

            for w in range(zemax_system.SystemData.Wavelengths.NumberOfWavelengths):

                rays_w = []

                for fi in range(field_positions_per_axis[0]):

                    fx = field_list_x[fi]

                    rays_fx = []

                    for fj in range(field_positions_per_axis[1]):

                        fy = field_list_y[fj]

                        is_field_inside_unit_circle = ((fx * fx) + (fy * fy)) <= 1.0

                        if not is_field_inside_unit_circle:
                            continue

                        rays_fy = []

                        for pi in range(pupil_positions_per_axis[0]):

                            px = pupil_list_x[pi]

                            rays_px = []

                            for pj in range(pupil_positions_per_axis[1]):

                                py = pupil_list_y[pj]

                                rays_py = [intensity]

                                rays_px.append([py, rays_py])

                            rays_fy.append([px, rays_px])

                        rays_fx.append([fy, rays_fy])

                    rays_w.append([fx, rays_fx])

                rays_s.append([w, rays_w])

            rays_c.append([s, rays_s])

        rays.append([c, rays_c])

    return trace(zemax_system, rays)


def trace(zemax_system: ZOSAPI.IOpticalSystem,
          rays: RayType,
          ) -> RayType:

    old_config = zemax_system.MCE.CurrentConfiguration

    result = []

    for c, rays_c in rays:

        result_c = []

        zemax_system.MCE.SetCurrentConfiguration(c + 1)

        # Initialize raytrace
        rt = zemax_system.Tools.OpenBatchRayTrace()  # raytrace object
        tool = zemax_system.Tools.CurrentTool  # pointer to active tool

        for s, rays_s in rays_c:

            result_s = []

            # Open instance of batch raytrace
            rt_dat = rt.CreateNormUnpol(100*100, ZOSAPI.Tools.RayTrace.RaysType.Real, s + 1)

            for w, rays_w in rays_s:

                result_w = []

                for fx, rays_fx in rays_w:

                    result_fx = []

                    for fy, rays_fy in rays_fx:

                        result_fy = []

                        rt_dat.ClearData()

                        for px, rays_px in rays_fy:

                            for py, rays_py in rays_px:

                                intensity, = rays_py

                                # Write ray to pipe
                                rt_dat.AddRay(w + 1, fx, fy, px, py, ZOSAPI.Tools.RayTrace.OPDMode.None_)

                        # Execute the raytrace
                        tool.RunAndWaitForCompletion()

                        # Initialize the process of reading the results of the raytrace
                        rt_dat.StartReadingResults()

                        for px, rays_px in rays_fy:

                            result_px = []

                            for py, rays_py in rays_px:

                                # Read next result from pipe
                                ret, n, err, vig, x, y, z, l, m, n, l2, m2, n2, opd, intensity = rt_dat.ReadNextResult()

                                if err != 0:
                                    continue

                                if vig != 0:
                                    continue

                                x *= u.mm
                                y *= u.mm

                                result_py = rays_py + [x, y]

                                result_px.append([py, result_py])

                            result_fy.append([px, result_px])

                        result_fx.append([fy, result_fy])

                    result_w.append([fx, result_fx])

                result_s.append([w, result_w])

            result_c.append([s, result_s])

        result.append([c, result_c])

        tool.Close()

    zemax_system.MCE.SetCurrentConfiguration(old_config)

    return result


def histogram(rays: RayType, bins_per_axis: int) -> Hist1DType:

    hist = []
    edges = []

    for c, rays_c in rays:

        hist_c = []
        edges_c = []

        for s, rays_s in rays_c:

            hist_s = []
            edges_s = []

            for w, rays_w in rays_s:

                x = []

                for fx, rays_fx in rays_w:

                    for fy, rays_fy in rays_fx:

                        for px, rays_px in rays_fy:

                            for py, rays_py in rays_px:

                                x.append(rays_py[-2])

                x = u.Quantity(x)
                h, e = np.histogram(x, bins=bins_per_axis)

                e *= x.unit

                hist_s.append(h)
                edges_s.append(e)

            hist_c.append(hist_s)
            edges_c.append(u.Quantity(edges_s))

        hist.append(hist_c)
        edges.append(u.Quantity(edges_c))

    hist = np.array(hist)
    edges = u.Quantity(edges)

    return hist, edges


def histogram_2d(rays: RayType, bins_per_axis: tp.Union[int, tp.Tuple[int, int]]) -> Hist2DType:

    if isinstance(bins_per_axis, int):
        bins_per_axis = (bins_per_axis, bins_per_axis)

    hist = []
    edges_x = []
    edges_y = []

    for c, rays_c in rays:

        hist_c = []
        edges_x_c = []
        edges_y_c = []

        for s, rays_s in rays_c:

            hist_s = []
            edges_x_s = []
            edges_y_s = []

            for w, rays_w in rays_s:

                x = []
                y = []

                for fx, rays_fx in rays_w:

                    for fy, rays_fy in rays_fx:

                        for px, rays_px in rays_fy:

                            for py, rays_py in rays_px:

                                x.append(rays_py[-2])
                                y.append(rays_py[-1])

                x = u.Quantity(x)
                y = u.Quantity(y)

                h, ex, ey = np.histogram2d(x, y, bins_per_axis)

                hist_s.append(h)
                edges_x_s.append(ex)
                edges_y_s.append(ey)

            hist_c.append(hist_s)
            edges_x_c.append(edges_x_s)
            edges_y_c.append(edges_y_s)

        hist.append(hist_c)
        edges_x.append(edges_x_c)
        edges_y.append(edges_y_c)

    hist = np.array(hist)
    edges_x = np.array(edges_x)
    edges_y = np.array(edges_y)

    return hist, edges_x, edges_y


def plot_histogram(hist1d: Hist1DType):

    with astropy.visualization.quantity_support():

        plt.figure()

        hist, edges = hist1d

        s_h = hist.shape
        s_x = edges.shape

        hist = hist.reshape((np.prod(s_h[:-1]), s_h[-1]))
        edges = edges.reshape((np.prod(s_x[:-1]), s_x[-1]))

        plt.step(edges[:, :-1].T, hist.T)


def plot_histogram_2d(hist2d: Hist2DType):

    plt.figure()

    hist, edges_x, edges_y = hist2d

    for hist_c, edges_x_c, edges_y_c in zip(hist, edges_x, edges_y):

        for hist_s, edges_x_s, edges_y_s in zip(hist_c, edges_x_c, edges_y_c):

            for hist_w, edges_x_w, edges_y_w in zip(hist_s, edges_x_s, edges_y_s):

                hist_w = hist_w.transpose()

                x_range = edges_x_w[-1] - edges_x_w[0]
                y_range = edges_y_w[-1] - edges_y_w[0]
                aspect = x_range / y_range

                extent = [edges_x_w[0], edges_x_w[-1], edges_y_w[0], edges_y_w[-1]]

                plt.figure()

                plt.imshow(hist_w, extent=extent, origin='lower', aspect=aspect)






