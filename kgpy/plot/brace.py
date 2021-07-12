import typing as typ
import numpy as np
import matplotlib.axes
import matplotlib.lines
import matplotlib.text

__all__ = ['vertical']


def vertical(
        ax: matplotlib.axes.Axes,
        x: float,
        ymin: float,
        ymax: float,
        width: float,
        text: str,
        beta: float = 300,
        resolution: int = 1001,
) -> typ.Tuple[matplotlib.lines.Line2D, matplotlib.text.Text]:

    y = np.linspace(ymin, ymax, resolution)
    y_half = y[:int(resolution / 2) + 1]
    y_half_brace = (1 / (1. + np.exp(-beta * (y_half - y_half[0]))) + 1 / (1. + np.exp(-beta * (y_half - y_half[-1]))))
    f = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    f = f - 0.5
    f = x + width * f

    if width > 0:
        ha = 'left'
        s = ' ' + text
    else:
        ha = 'right'
        s = text + ' '

    line, = ax.plot(f, y, color='black', lw=1, clip_on=False, solid_capstyle='round')

    text, = ax.text(
        x=x + 1.2 * width,
        y=(ymax + ymin) / 2.,
        s=s,
        ha=ha,
        va='center',
    )

    return line, text
