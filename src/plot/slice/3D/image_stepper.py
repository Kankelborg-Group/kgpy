import numpy
import matplotlib.pyplot as plt


class ImageSlicer(object):
    def __init__(self, X):
        fig, ax = plt.subplots(1, 1)
        self.fig = fig
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = 0

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = numpy.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = numpy.clip(self.ind - 1, 0, self.slices - 1)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def test_ImageSlicer():

    X = numpy.random.rand(20, 20, 40)
    img = ImageSlicer(X)
    plt.show()

if __name__ == '__main__':
    test_ImageSlicer()