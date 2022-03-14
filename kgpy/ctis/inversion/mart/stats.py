import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from kgpy.moment.percentile import arg_percentile

__all__ = ['histogram_tetraptych']


def histogram_tetraptych(
        x: typ.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        y: typ.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        percentile_threshold_x: float = .1,
        percentile_threshold_y: float = 1.,
        num_histogram_bins: int = 100,
        x_axis_labels: typ.Tuple[str, str, str, str] = ('', '', '', ''),
        y_axis_labels: typ.Tuple[str, str, str, str] = ('', '', '', ''),
        red_line: bool = False,
        interquartile_contour: bool = False,
        min_iqr_count: int = 10,
) -> plt.Figure:
    """
    Produces a figure with four subplots, each a 2D histogram between corresponding entries in `x` and `y`

    :param x: tuple of 4 arrays
    :param y: tuple of 4 arrays
    :param percentile_threshold_x: parameter used to calculate max and min percentile values for arrays in `x`
    :param percentile_threshold_y: parameter used to calculate max and min percentile values for arrays in `y`
    :param num_histogram_bins: how many bins in each the x and y axes for the histogram
    :param x_axis_labels:
    :param y_axis_labels:
    :param red_line: if `True`, plot the "x=y" line in each subplot as a red line
    :param interquartile_contour: if `True`, plots contour of 25th, 50th, and 75th percentile.
    :param min_iqr_count: minimum number of unique values in a column for that column to be considered in calculating
        the interquartile contour
    :return:
    """
    # Makes assumption that these are already masked!
    x0, x1, x2, x3 = x
    y0, y1, y2, y3 = y

    ht_lower_x = percentile_threshold_x
    ht_upper_x = 100 - percentile_threshold_x
    ht_lower_y = percentile_threshold_y
    ht_upper_y = 100 - percentile_threshold_y

    x_ranges = [(np.nanpercentile(xj, ht_lower_x), np.nanpercentile(xj, ht_upper_x)) for xj in [x0, x1, x2, x3]]
    y_ranges = [(np.nanpercentile(yj, ht_lower_y), np.nanpercentile(yj, ht_upper_y)) for yj in [y0, y1, y2, y3]]

    fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    # colornorm = matplotlib.colors.SymLogNorm(1e-3)
    colornorm = None

    histos = []
    imgs = []
    flat_ax = (ax.flatten())
    edgepairs = []

    for j in range(len(x_ranges)):
        # index choices
        xj = x[j]
        yj = y[j]
        x_range_j = x_ranges[j]
        y_range_j = y_ranges[j]

        hist_j, x_edges_j, y_edges_j = np.histogram2d(xj, yj, bins=num_histogram_bins, range=(x_range_j, y_range_j))
        hist_j /= np.max(hist_j, axis=~0, keepdims=True)
        hist_j[np.isnan(hist_j)] = 0
        histos.append(hist_j)

        img_j = flat_ax[j].imshow(hist_j.T, norm=colornorm, extent=(x_range_j + y_range_j), origin='lower')
        flat_ax[j].set_xlabel(x_axis_labels[j])
        flat_ax[j].set_ylabel(y_axis_labels[j])
        fig.colorbar(img_j, ax=flat_ax[j])
        imgs.append(img_j)

        edgepairs.append(
            (x_edges_j, y_edges_j)
        )

        if red_line:
            """
            Want red line to be on y=x through the histogram plot. The line y=x may not necessarily be through the
            middle diagonal of the plot. Need to grab edges, do some calculations to see where the y=x line hits the
            edges of the histogram plot.
            """
            x_min = x_edges_j[0]
            x_max = x_edges_j[-1]
            y_min = y_edges_j[0]
            y_max = y_edges_j[-1]

            # find the correct "entrance" point, i.e. where the y=x line enters on the bottom-left side
            if y_min <= x_min:
                entrance_value = x_min
            else:
                # where y_min > x_min
                entrance_value = y_min

            # find the the correct "exit" point, i.e. where the y=x line exits on the top-right side
            if y_max <= x_max:
                exit_value = y_max
            else:
                # where y_max > x_max
                exit_value = x_max

            red_line_range = (entrance_value, exit_value)
            # Plot it
            flat_ax[j].plot(red_line_range, red_line_range, color='red', linewidth=1.5)

        if interquartile_contour:
            # only interested in columns that have enough data values that IQR is useful
            number_of_columns = hist_j.shape[-1]
            unique_value_counts = np.zeros(number_of_columns)
            for m in range(number_of_columns):
                unique_value_counts[m] = (np.nonzero(hist_j[:, m])[0]).size

            columns_to_keep = (unique_value_counts > min_iqr_count)

            percentiles = [.25, .50, .75]

            for p in percentiles:
                # grab indices of this percentile
                idx = np.floor(arg_percentile(hist_j, p)).flatten()
                # locate infs and NaNs so they can be reomoved
                infs_or_nan = np.logical_or(np.isnan(idx), np.isinf(idx))
                good_idx = np.logical_and(~infs_or_nan, columns_to_keep)
                idx = (idx[good_idx]).astype(np.int).flatten()

                # x-array needs corresponding values eliminated too
                x_keep = x_edges_j[:~0][good_idx]
                if p == 0.50:
                    flat_ax[j].plot(x_keep, y_edges_j[idx], '-', color='orange')
                else:
                    flat_ax[j].plot(x_keep, y_edges_j[idx], '-', color='white')

    for axis in flat_ax:
        axis.set_aspect('auto')

    return fig


def multi_tetraptych(
        independent: typ.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        dependents: typ.List[typ.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        percentile_threshold_x: float = .1,
        percentile_threshold_y: float = 1.,
        num_histogram_bins: int = 100,
        x_axis_labels: typ.Tuple[str, str, str, str] = ('', '', '', ''),
        y_axis_labels: typ.Tuple[str, str, str, str] = ('', '', '', ''),
        dependents_labels: typ.Tuple[str, ...] = ('', ''),
        red_line: bool = False,
        contour: str = 'iqr',
        min_iqr_count: int = 10,
):
    """
    Produces a figure with four subplots, each a 2D histogram between corresponding entries in `x` and `y`

    :param independent: tuple of 4 arrays, with moments to be along x-dimension of historam plots
    :param dependents: list of tuples of 4 arrays
    :param percentile_threshold_x: parameter used to calculate max and min percentile values for arrays in `x`
    :param percentile_threshold_y: parameter used to calculate max and min percentile values for arrays in `y`
    :param num_histogram_bins: how many bins in each the x and y axes for the histogram
    :param x_axis_labels:
    :param y_axis_labels:
    :param dependents_labels: list of strings to be used in legends on plots.
    :param red_line: if `True`, plot the "x=y" line in each subplot as a red line
    :param contour: if `True`, plots contour of 25th, 50th, and 75th percentile.
    :param min_iqr_count: minimum number of unique values in a column for that column to be considered in calculating
        the interquartile contour
    :return:
    """
    # for correctly specifying color:
    color = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'olive', 'cyan']

    if dependents_labels == ('', ''):
        dependents_labels == tuple(['Input {}'.format(m) for m in range(len(dependents))])

    # make assumption that input arrays are already masked
    x0, x1, x2, x3 = independent

    # Thresholds
    ht_lower_x = percentile_threshold_x
    ht_upper_x = 100 - percentile_threshold_x
    ht_lower_y = percentile_threshold_y
    ht_upper_y = 100 - percentile_threshold_y

    x_ranges = [(np.nanpercentile(xj, ht_lower_x), np.nanpercentile(xj, ht_upper_x)) for xj in [x0, x1, x2, x3]]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), dpi=200, constrained_layout=True)
    # colornorm = matplotlib.colors.SymLogNorm(1e-3)
    colornorm = None

    histos = []
    imgs = []
    flat_ax = (ax.flatten())
    edgepairs = []

    for k, dep_mom in enumerate(dependents):
        y0, y1, y2, y3 = dep_mom
        y_ranges = [(np.nanpercentile(yk, ht_lower_y), np.nanpercentile(yk, ht_upper_y)) for yk in [y0, y1, y2, y3]]

        for j in range(len(x_ranges)):
            # index choices
            xj = independent[j]
            yj = dep_mom[j]
            x_range_j = x_ranges[j]
            y_range_j = y_ranges[j]

            hist_j, x_edges_j, y_edges_j = np.histogram2d(xj, yj, bins=num_histogram_bins, range=(x_range_j, y_range_j))
            hist_j /= np.max(hist_j, axis=~0, keepdims=True)
            hist_j[np.isnan(hist_j)] = 0
            histos.append(hist_j)

            # img_j = flat_ax[j].imshow(hist_j.T, norm=colornorm, extent=(x_range_j + y_range_j), origin='lower')
            flat_ax[j].set_xlabel(x_axis_labels[j])
            flat_ax[j].set_ylabel(y_axis_labels[j])
            # fig.colorbar(img_j, ax=flat_ax[j])
            # imgs.append(img_j)

            edgepairs.append(
                (x_edges_j, y_edges_j)
            )

            # if red_line and k == 0:
            #     """
            #     Want red line to be on y=x through the histogram plot. The line y=x may not necessarily be through the
            #     middle diagonal of the plot. Need to grab edges, do some calculations to see where the y=x line hits the
            #     edges of the histogram plot.
            #     """
            #     x_min = x_edges_j[0]
            #     x_max = x_edges_j[-1]
            #     y_min = y_edges_j[0]
            #     y_max = y_edges_j[-1]
            #
            #     # find the correct "entrance" point, i.e. where the y=x line enters on the bottom-left side
            #     if y_min <= x_min:
            #         entrance_value = x_min
            #     else:
            #         # where y_min > x_min
            #         entrance_value = y_min
            #
            #     # find the the correct "exit" point, i.e. where the y=x line exits on the top-right side
            #     if y_max <= x_max:
            #         exit_value = y_max
            #     else:
            #         # where y_max > x_max
            #         exit_value = x_max
            #
            #     red_line_range = (entrance_value, exit_value)
            #     # Plot it
            #     flat_ax[j].plot(red_line_range, red_line_range, color='red', linewidth=.8)

            if contour is not None:
                # only interested in columns that have enough data values that IQR is useful
                number_of_columns = hist_j.shape[-1]
                unique_value_counts = np.zeros(number_of_columns)
                for m in range(number_of_columns):
                    unique_value_counts[m] = (np.nonzero(hist_j[:, m])[0]).size

                columns_to_keep = (unique_value_counts > min_iqr_count)

                if contour == 'iqr':
                    percentiles = [.25, .50, .75]
                elif contour == 'median':
                    percentiles = [.50]

                for p in percentiles:
                    # grab indices of this percentile
                    idx = np.floor(arg_percentile(hist_j, p)).flatten()
                    # locate infs and NaNs so they can be reomoved
                    infs_or_nan = np.logical_or(np.isnan(idx), np.isinf(idx))
                    good_idx = np.logical_and(~infs_or_nan, columns_to_keep)
                    idx = (idx[good_idx]).astype(np.int).flatten()

                    # x-array needs corresponding values eliminated too
                    x_keep = x_edges_j[:~0][good_idx]
                    if p == 0.50:
                        flat_ax[j].plot(x_keep, y_edges_j[idx], '-', color=color[k], label=dependents_labels[k])
                    else:
                        flat_ax[j].plot(x_keep, y_edges_j[idx], '-', color=color[k])

    if red_line:
        for k in range(4):
            x_min = x_ranges[k][0]
            x_max = x_ranges[k][1]
            # y_min = min([dep[k].min() for dep in dependents])
            # y_max = max([dep[k].max() for dep in dependents])
            xarr = np.asarray([x_min, x_max])
            flat_ax[k].plot(xarr, xarr, color='red')

    for ax in flat_ax:
        ax.set_aspect('auto')
        ax.legend()

    return fig
