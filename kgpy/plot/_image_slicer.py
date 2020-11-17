import numpy as np
import matplotlib.pyplot as plt

__all__ = ['ImageSlicer']


class ImageSlicer:

    def __init__(self, x, y, **kwargs):

        fig, ax = plt.subplots(1, 1)
        self.fig = fig
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        x, y = np.broadcast_arrays(x, y, subok=True)
        self.x = x
        self.y = y
        self.sh = x.shape
        self.ind = 0

        self.plot = ax.plot(self.x[self.ind], self.y[self.ind], **kwargs)
        self.update()

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.sh[0] - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.sh[0] - 1)
        self.update()

    def update(self):

        self.ax.set_ylabel('slice %s' % self.ind)

        self.plot[0].set_xdata(self.x[self.ind])
        self.plot[0].set_ydata(self.y[self.ind])

        self.fig.canvas.draw()
