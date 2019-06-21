import numpy
import matplotlib.pyplot as plt
import os


class CubeSlicer(object):
    def __init__(self, X, y=None, **kwargs):
        fig, ax = plt.subplots(1, 1)
        self.fig = fig
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.sh = X.shape
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind,], **kwargs)
        self.update()

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = numpy.clip(self.ind + 1, 0, self.sh[0] - 1)
        else:
            self.ind = numpy.clip(self.ind - 1, 0, self.sh[0] - 1)
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind,])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def set_data(self, X):
        self.X = X
        self.ind = 0
        self.update()

    def save(self,path: str):
        """
        Method defined to save each image in the cube.  Images are saved in directory "path" in order as i.png
        :param path:
        :return:
        """
        j = self.ind
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(self.slices):
            self.ind = i
            self.update()
            self.fig.savefig(os.path.join(os.path.dirname(path),str(i)+'.png'))
        self.ind = j

def test_CubeSlicer():

    X = numpy.random.rand(20, 20, 40)
    Y = numpy.random.rand(20, 20, 40)
    img = CubeSlicer(X)
    img2 = CubeSlicer(Y)
    plt.show()

if __name__ == '__main__':
    test_CubeSlicer()