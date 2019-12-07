import numpy
import matplotlib.pyplot as plt

__all__ = ['ImageSlicer']


class ImageSlicer:

    def __init__(self, x, y, **kwargs):

        fig, ax = plt.subplots(1, 1)
        self.fig = fig
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.x = x
        self.y = y
        self.sh = x.shape
        self.ind = 0

        self.plot = ax.plot(self.x[self.ind], self.y[self.ind], **kwargs)
        self.update()

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = numpy.clip(self.ind + 1, 0, self.sh[0] - 1)
        else:
            self.ind = numpy.clip(self.ind - 1, 0, self.sh[0] - 1)
        self.update()

    def update(self):

        self.plot.set_xdata(self.x[self.ind])
        self.plot.set_ydata(self.y[self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.plot.axes.figure.canvas.draw()
