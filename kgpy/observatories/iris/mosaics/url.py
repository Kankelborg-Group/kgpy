import pathlib
import urlpath

__all__ = ['path_list']


base = urlpath.URL('https://www.lmsal.com/solarsoft/irisa/data/level2_compressed')


path_list = [
    pathlib.Path('2013/09/30/20130930Mosaic/IRISMosaic_20130930_Si1393.fits.gz'),
    pathlib.Path('2013/10/13/20131013Mosaic/IRISMosaic_20131013_Si1393.fits.gz'),
    pathlib.Path('2013/10/21/20131021Mosaic/IRISMosaic_20131021_Si1393.fits.gz'),
    pathlib.Path('2013/10/27/20131027Mosaic/IRISMosaic_20131027_Si1393.fits.gz'),
    pathlib.Path('2014/03/17/20140317Mosaic/IRISMosaic_20140317_Si1393.fits.gz'),
    pathlib.Path('2014/03/24/20140324Mosaic/IRISMosaic_20140324_Si1393.fits.gz'),
    pathlib.Path('2014/05/12/20140512Mosaic/IRISMosaic_20140512_Si1393.fits.gz'),
    pathlib.Path('2014/05/27/20140527Mosaic/IRISMosaic_20140527_Si1393.fits.gz'),
    pathlib.Path('2014/06/23/20140623Mosaic/IRISMosaic_20140623_Si1393.fits.gz'),
    pathlib.Path('2014/07/27/20140727Mosaic/IRISMosaic_20140727_Si1393.fits.gz'),
    pathlib.Path('2014/08/24/20140824Mosaic/IRISMosaic_20140824_Si1393.fits.gz'),
    pathlib.Path('2014/09/21/20140921Mosaic/IRISMosaic_20140921_Si1393.fits.gz'),
    pathlib.Path('2014/10/14/20141014Mosaic/IRISMosaic_20141014_Si1393.fits.gz'),
    pathlib.Path('2014/10/20/20141020Mosaic/IRISMosaic_20141020_Si1393.fits.gz'),
    pathlib.Path('2015/02/22/20150222Mosaic/IRISMosaic_20150222_Si1393.fits.gz'),
    pathlib.Path('2015/03/01/20150301Mosaic/IRISMosaic_20150301_Si1393.fits.gz'),
    pathlib.Path('2015/03/22/20150322Mosaic/IRISMosaic_20150322_Si1393.fits.gz'),
    pathlib.Path('2015/04/01/20150401Mosaic/IRISMosaic_20150401_Si1393.fits.gz'),
    pathlib.Path('2015/04/27/20150427Mosaic/IRISMosaic_20150427_Si1393.fits.gz'),
    pathlib.Path('2015/05/31/20150531Mosaic/IRISMosaic_20150531_Si1393.fits.gz'),
    pathlib.Path('2015/06/29/20150629Mosaic/IRISMosaic_20150629_Si1393.fits.gz'),
    pathlib.Path('2015/07/26/20150726Mosaic/IRISMosaic_20150726_Si1393.fits.gz'),
    pathlib.Path('2015/08/23/20150823Mosaic/IRISMosaic_20150823_Si1393.fits.gz'),
    pathlib.Path('2015/09/27/20150927Mosaic/IRISMosaic_20150927_Si1393.fits.gz'),
    pathlib.Path('2015/10/12/20151012Mosaic/IRISMosaic_20151012_Si1393.fits.gz'),
    pathlib.Path('2015/10/18/20151018Mosaic/IRISMosaic_20151018_Si1393.fits.gz'),
    pathlib.Path('2015/10/27/20151027Mosaic/IRISMosaic_20151027_Si1393.fits.gz'),
    pathlib.Path('2016/02/22/20160222Mosaic/IRISMosaic_20160222_Si1393.fits.gz'),
    pathlib.Path('2016/03/28/20160328Mosaic/IRISMosaic_20160328_Si1393.fits.gz'),
    pathlib.Path('2016/04/25/20160425Mosaic/IRISMosaic_20160425_Si1393.fits.gz'),
    pathlib.Path('2016/05/01/20160501Mosaic/IRISMosaic_20160501_Si1393.fits.gz'),
    pathlib.Path('2016/05/22/20160522Mosaic/IRISMosaic_20160522_Si1393.fits.gz'),
    pathlib.Path('2016/07/05/20160705Mosaic/IRISMosaic_20160705_Si1393.fits.gz'),
    pathlib.Path('2016/07/31/20160731Mosaic/IRISMosaic_20160731_Si1393.fits.gz'),
    pathlib.Path('2016/08/22/20160822Mosaic/IRISMosaic_20160822_Si1393.fits.gz'),
    pathlib.Path('2016/09/25/20160925Mosaic/IRISMosaic_20160925_Si1393.fits.gz'),
    pathlib.Path('2016/10/09/20161009Mosaic/IRISMosaic_20161009_Si1393.fits.gz'),
    pathlib.Path('2016/10/17/20161017Mosaic/IRISMosaic_20161017_Si1393.fits.gz'),
    pathlib.Path('2016/10/24/20161024Mosaic/IRISMosaic_20161024_Si1393.fits.gz'),
    pathlib.Path('2017/02/27/20170227Mosaic/IRISMosaic_20170227_Si1393.fits.gz'),
    pathlib.Path('2017/03/12/20170312Mosaic/IRISMosaic_20170312_Si1393.fits.gz'),
    pathlib.Path('2017/03/27/20170327Mosaic/IRISMosaic_20170327_Si1393.fits.gz'),
    pathlib.Path('2017/04/24/20170424Mosaic/IRISMosaic_20170424_Si1393.fits.gz'),
    pathlib.Path('2017/05/21/20170521Mosaic/IRISMosaic_20170521_Si1393.fits.gz'),
    pathlib.Path('2017/06/26/20170626Mosaic/IRISMosaic_20170626_Si1393.fits.gz'),
    # pathlib.Path('2017/07/24/20170724Mosaic/IRISMosaic_20170724_Si1393.fits.gz'), this one is zero size currently
    pathlib.Path('2017/08/28/20170828Mosaic/IRISMosaic_20170828_Si1393.fits.gz'),
    pathlib.Path('2017/09/24/20170924Mosaic/IRISMosaic_20170924_Si1393.fits.gz'),
    pathlib.Path('2017/10/16/20171016Mosaic/IRISMosaic_20171016_Si1393.fits.gz'),
    pathlib.Path('2017/10/21/20171021Mosaic/IRISMosaic_20171021_Si1393.fits.gz'),
    # pathlib.Path('2018/02/26/20180226Mosaic/IRISMosaic_20180226_Si1393.fits.gz'), gzip error
    pathlib.Path('2018/03/12/20180312Mosaic/IRISMosaic_20180312_Si1393.fits.gz'),
    pathlib.Path('2018/03/26/20180326Mosaic/IRISMosaic_20180326_Si1393.fits.gz'),
    pathlib.Path('2018/04/22/20180422Mosaic/IRISMosaic_20180422_Si1393.fits.gz'),
    pathlib.Path('2018/05/27/20180527Mosaic/IRISMosaic_20180527_Si1393.fits.gz'),
    pathlib.Path('2018/06/25/20180625Mosaic/IRISMosaic_20180625_Si1393.fits.gz'),
    pathlib.Path('2018/07/23/20180723Mosaic/IRISMosaic_20180723_Si1393.fits.gz'),
    pathlib.Path('2018/08/25/20180825Mosaic/IRISMosaic_20180825_Si1393.fits.gz'),
]
