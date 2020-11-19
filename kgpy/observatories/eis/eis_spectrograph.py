import copy

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from ndcube import NDCube, NDCubeSequence
from ndcube.utils.wcs import WCS
from ndcube.utils.cube import convert_extra_coords_dict_to_input_format



__all__ = ['EISSpectrograph']

class EISSpectrograph(object):
    """
    An object to hold data from multiple EIS raster scans.

    """

    def __init__(self, data, meta=None):
        self.data = data
        self.meta = meta


    def __repr__(self):
        return('EISSpectrograph Object, modeled from IRISPY IRISSpectrograph')



class EISSpectrogramCubeSequence(NDCubeSequence):
    """Class for holding, slicing and plotting EIS spectrogram data.

    This class contains all the functionality of its super class with
    some additional functionalities.

    Parameters
    ----------
    data_list: `list`
        List of `EISSpectrogramCube` objects from the same spectral window and OBS ID.
        Must also contain the 'detector type' in its meta attribute.

    meta: `dict` or header object
        Metadata associated with the sequence.

    common_axis: `int`
        The axis of the NDCubes corresponding to time.

    """
    def __init__(self, data_list, meta=None, common_axis=0):
        # detector_type_key = "detector type"
        # # Check that meta contains required keys.
        # required_meta_keys = [detector_type_key, "spectral window",
        #                       "brightest wavelength", "min wavelength", "max wavelength"]
        # if not all([key in list(meta) for key in required_meta_keys]):
        #     raise ValueError("Meta must contain following keys: {0}".format(required_meta_keys))
        # # Check that all spectrograms are from same specral window and OBS ID.
        # if len(np.unique([cube.meta["OBSID"] for cube in data_list])) != 1:
        #     raise ValueError("Constituent IRISSpectrogramCube objects must have same "
        #                      "value of 'OBSID' in its meta.")
        # if len(np.unique([cube.meta["spectral window"] for cube in data_list])) != 1:
        #     raise ValueError("Constituent IRISSpectrogramCube objects must have same "
        #                      "value of 'spectral window' in its meta.")
        # Initialize Sequence.
        super(EISSpectrogramCubeSequence, self).__init__(
            data_list, meta=meta, common_axis=common_axis)



class EISSpectrogramCube(NDCube):
    """
    Class representing IRISSpectrogramCube data described by a single WCS.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.

    wcs: `ndcube.wcs.wcs.WCS`
        The WCS object containing the axes' information

    unit : `astropy.unit.Unit` or `str`
        Unit for the dataset. Strings that can be converted to a Unit are allowed.

    meta : dict-like object
        Additional meta information about the dataset. Must contain at least the
        following keys:
            detector type: str, (FUV1, FUV2 or NUV)
            OBSID: int
            spectral window: str

    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn’t mandatory. If the uncertainty
        has no such attribute the uncertainty is stored as UnknownUncertainty.
        Defaults to None.

    mask : any type, optional
        Mask for the dataset. Masks should follow the numpy convention
        that valid data points are marked by False and invalid ones with True.
        Defaults to None.

    extra_coords : iterable of `tuple`s, each with three entries
        (`str`, `int`, `astropy.units.quantity` or array-like)
        Gives the name, axis of data, and values of coordinates of a data axis not
        included in the WCS object.

    copy : `bool`, optional
        Indicates whether to save the arguments as copy. True copies every attribute
        before saving it while False tries to save every parameter as reference.
        Note however that it is not always possible to save the input as reference.
        Default is False.
    """

    def __init__(self, data, wcs, meta = None,
                 mask=None, copy=False, missing_axes=None):
        # unit,  extra_coords,

        # # Check required meta data is provided.
        # required_meta_keys = ["detector type"]
        # if not all([key in list(meta) for key in required_meta_keys]):
        #         raise ValueError("Meta must contain following keys: {0}".format(required_meta_keys))
        # # Check extra_coords contains required coords.
        # required_extra_coords_keys = ["time", "exposure time"]
        # extra_coords_keys = [coord[0] for coord in extra_coords]
        # if not all([key in extra_coords_keys for key in required_extra_coords_keys]):
        #     raise ValueError("The following extra coords must be supplied: {0} vs. {1} from {2}".format(
        #         required_extra_coords_keys, extra_coords_keys, extra_coords))

        # Initialize EISSpectrogramCube.
        super(EISSpectrogramCube, self).__init__(
            data, wcs, meta = meta)

    def __getitem__(self, item):
        result = super(EISSpectrogramCube, self).__getitem__(item)
        return EISSpectrogramCube(
            result.data, result.wcs)
            # result.uncertainty, result.unit, result.meta,
            # convert_extra_coords_dict_to_input_format(result.extra_coords, result.missing_axes),
            # mask=result.mask, missing_axes=result.missing_axes)




def read_eis_spectrograph_level1_fits(filenames, spectral_windows=None, uncertainty=True, memmap=False):
    """
    #
    # Parameters
    # ----------
    # filenames: `list` of `str` or `str`
    #     Filename of filenames to be read.  They must all be associated with the same
    #     OBS number.
    #
    # spectral_windows: iterable of `str` or `str`
    #     Spectral windows to extract from files.  Default=None, implies, extract all
    #     spectral windows.
    #

    """
    if type(filenames) is str:
        filenames = [filenames]
    for f, filename in enumerate(filenames):
        hdulist = fits.open(filename, memmap=memmap, do_not_scale_image_data=memmap)
        hdulist.verify('fix')
        if f == 0:
            # Determine number of raster positions in a scan
            raster_positions_per_scan = int(hdulist[0].header["NRASTER"])
            # Collecting the window observations.
            windows_in_obs = np.array([hdulist[1].header["TTYPE{0}".format(i)]
                                       for i in range(1, hdulist[0].header["NWIN"]+1)])

            # If spectral_window is not set then get every window.
            # Else take the appropriate windows
            if not spectral_windows:
                spectral_windows_req = windows_in_obs
                window_fits_indices = range(1, hdulist[0].header["NWIN"]+1)
            else:
                if type(spectral_windows) is str:
                    spectral_windows_req = [spectral_windows]
                else:
                    spectral_windows_req = spectral_windows
                spectral_windows_req = np.asarray(spectral_windows_req, dtype="U")
                window_is_in_obs = np.asarray(
                    [window in windows_in_obs for window in spectral_windows_req])
                if not all(window_is_in_obs):
                    missing_windows = window_is_in_obs == False
                    raise ValueError("Spectral windows {0} not in file {1}".format(
                        spectral_windows[missing_windows], filenames[0]))
                window_fits_indices = np.nonzero(np.in1d(windows_in_obs,
                                                         spectral_windows))[0]+1
            # Generate top level meta dictionary from first file
            # main header.
            top_meta = {"TELESCOP": hdulist[0].header["TELESCOP"],
                        "INSTRUME": hdulist[0].header["INSTRUME"],
                        "DATA_LEV": hdulist[0].header["DATA_LEV"],
                        "OBSTITLE": hdulist[0].header["OBSTITLE"],
                        "OBS_DESC": hdulist[0].header["OBS_DEC"],
                        "DATE_OBS": Time(hdulist[0].header["DATE_OBS"]),
                        "DATE_END": Time(hdulist[0].header["DATE_END"]),
                        "SAT_ROT": hdulist[0].header["SAT_ROT"] * u.deg,
                        "FOVX": hdulist[0].header["FOVX"] * u.arcsec,
                        "FOVY": hdulist[0].header["FOVY"] * u.arcsec,
                        # "NEXPOBS": hdulist[0].header["NEXPOBS"],
                        "NRASTER": hdulist[0].header["NRASTER"]}

            # Initialize meta dictionary for each spectral_window
            window_metas = {}

            for i, window_name in enumerate(spectral_windows_req):

                window_metas[window_name] = {
                    "spectral window":
                        hdulist[1].header["TTYPE{0}".format(window_fits_indices[i])],
                    "brightest wavelength":
                        hdulist[1].header["TWAVE{0}".format(window_fits_indices[i])],
                    "min wavelength":
                        hdulist[1].header["TWMIN{0}".format(window_fits_indices[i])],
                    "max wavelength":
                        hdulist[1].header["TWMAX{0}".format(window_fits_indices[i])],
                    "SAT_ROT": hdulist[0].header["SAT_ROT"],
                }
            # Create a empty list for every spectral window and each
            # spectral window is a key for the dictionary.
            data_dict = dict([(window_name, list())
                              for window_name in spectral_windows_req])


        # Determine extra coords for this raster.

        # times = (Time(hdulist[0].header["STARTOBS"]) +
        #          TimeDelta(hdulist[-2].data[:, hdulist[-2].header["TIME"]], format='sec'))
        # raster_positions = np.arange(int(hdulist[0].header["NRASTER"]))
        # pztx = hdulist[-2].data[:, hdulist[-2].header["PZTX"]] * u.arcsec
        # pzty = hdulist[-2].data[:, hdulist[-2].header["PZTY"]] * u.arcsec
        # xcenix = hdulist[-2].data[:, hdulist[-2].header["XCENIX"]] * u.arcsec
        # ycenix = hdulist[-2].data[:, hdulist[-2].header["YCENIX"]] * u.arcsec
        # obs_vrix = hdulist[-2].data[:, hdulist[-2].header["OBS_VRIX"]] * u.m/u.s
        # ophaseix = hdulist[-2].data[:, hdulist[-2].header["OPHASEIX"]]
        # exposure_times_fuv = hdulist[-2].data[:, hdulist[-2].header["EXPTIMEF"]] * u.s
        # exposure_times_nuv = hdulist[-2].data[:, hdulist[-2].header["EXPTIMEN"]] * u.s

        # If OBS is raster, include raster positions.  Otherwise don't.
        # if top_meta["NRASTER"] > 1:
        #     general_extra_coords = [("time", 0, times),
        #                             ("raster position", 0, np.arange(top_meta["NRASTERP"])),
        #                             ("pztx", 0, pztx), ("pzty", 0, pzty),
        #                             ("xcenix", 0, xcenix), ("ycenix", 0, ycenix),
        #                             ("obs_vrix", 0, obs_vrix), ("ophaseix", 0, ophaseix)]
        # else:
        #     general_extra_coords = [("time", 0, times),
        #                             ("pztx", 0, pztx), ("pzty", 0, pzty),
        #                             ("xcenix", 0, xcenix), ("ycenix", 0, ycenix),
        #                             ("obs_vrix", 0, obs_vrix), ("ophaseix", 0, ophaseix)]


        for i, window_name in enumerate(spectral_windows_req):
            # Derive WCS, data and mask for NDCube from file.
            # Rearrange ESIS data for each spectral window
            # print(hdulist[1].data.shape)
            data = np.array([hdulist[1].data[j][window_fits_indices[i]-1] for j in range(len(hdulist[1].data))])



            # Sit-and-stare have a CDELT of 0 which causes issues in astropy WCS.
            # In this case, set CDELT to a tiny non-zero number.
            # if hdulist[0].header["CDELT3"] == 0:
            #     hdulist[0].header["CDELT3"] = 1e-10

            #  The above correction may be needed in the future but CDELT3 does not correspond to Solar-X for EIS


            #EIS WCS is a complete mess so I think we need to go manual for now.
            hdr = hdulist[1].header

            #crval1 will need to be calculated from the spectral window info I think.  More to come.
            wcs_input_dict = {
                'CTYPE1': hdr['CTYPE3'], 'CUNIT1': hdr['CUNIT3'], 'CDELT1': hdr['CDELT3'], 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': len(data[0,0,:]),
                'CTYPE2': hdr['CTYPE2'], 'CUNIT2': hdr['CUNIT2'], 'CDELT2': hdr['CDELT2'], 'CRPIX2': hdr['CRPIX2'], 'CRVAL2': hdr['CRVAL2'], 'NAXIS2': len(data[0,:,0]),
                'CTYPE3': hdr['CTYPE1'], 'CUNIT3': hdr['CUNIT1'], 'CDELT3': hdr['CDELT1'], 'CRPIX3': hdr['CRPIX1'], 'CRVAL3': hdr['CRVAL1'], 'NAXIS3': len(data[:,0,0])}

            wcs_ = WCS(wcs_input_dict)



            # if not memmap:
            #     data_mask = data == -200.
            # else:
            #     data_mask = None


            # Derive extra coords for this spectral window.
            # window_extra_coords = copy.deepcopy(general_extra_coords)
            # window_extra_coords.append(("exposure time", 0, exposure_times))
            # Collect metadata relevant to single files.
            try:
                date_obs = Time(hdulist[0].header["DATE_OBS"])
            except ValueError:
                date_obs = None
            try:
                date_end = Time(hdulist[0].header["DATE_END"])
            except ValueError:
                date_end = None
            single_file_meta = {"SAT_ROT": hdulist[0].header["SAT_ROT"] * u.deg,
                                "DATE_OBS": date_obs,
                                "DATE_END": date_end,
                                # "HLZ": bool(int(hdulist[0].header["HLZ"])),
                                # "SAA": bool(int(hdulist[0].header["SAA"])),
                                # "DSUN_OBS": hdulist[0].header["DSUN_OBS"] * u.m,
                                # "IAECEVFL": hdulist[0].header["IAECEVFL"],
                                # "IAECFLAG": hdulist[0].header["IAECFLAG"],
                                # "IAECFLFL": hdulist[0].header["IAECFLFL"],
                                # "KEYWDDOC": hdulist[0].header["KEYWDDOC"],
                                # "detector type":
                                #      hdulist[0].header["TDET{0}".format(window_fits_indices[i])],
                                "spectral window": window_name,
                                # "OBSID": hdulist[0].header["OBSID"],
                                "OBS_DESC": hdulist[0].header["OBS_DEC"],
                                # "STARTOBS": Time(hdulist[0].header["STARTOBS"]),
                                # "ENDOBS": Time(hdulist[0].header["ENDOBS"])
                                }


            # Derive uncertainty of data
            # if uncertainty:
            #     out_uncertainty = u.Quantity(np.sqrt(
            #         (hdulist[window_fits_indices[i]].data*DN_unit).to(u.photon).value +
            #         readout_noise.to(u.photon).value**2), unit=u.photon).to(DN_unit).value
            # else:
            #     out_uncertainty = None
            # Appending NDCube instance to the corresponding window key in dictionary's list.
            data_dict[window_name].append(
                NDCube(data, wcs_,
                                meta = single_file_meta,))
        hdulist.close()

    # print(type(data_dict['Fe XII 186.750'][0].wcs))

    # Construct dictionary of IRISSpectrogramCubeSequences for spectral windows
    data = dict([(window_name, EISSpectrogramCubeSequence(data_dict[window_name],
                                                          meta=window_metas[window_name],
                                                          common_axis=0))
                                              for window_name in spectral_windows_req])
    # Initialize an EISSpectrograph object.
    return EISSpectrograph(data, meta=top_meta)



# def _produce_obs_repr_string(meta):
#     obs_info = [meta.get(key, "Unknown") for key in ["OBSID", "OBS_DESC", "STARTOBS", "ENDOBS"]]
#     return """OBS ID: {obs_id}
# OBS Description: {obs_desc}
# OBS period: {obs_start} -- {obs_end}""".format(obs_id=obs_info[0], obs_desc=obs_info[1],
#                                                obs_start=obs_info[2], obs_end=obs_info[3])
#
#
# def _try_parse_time_on_meta(meta):
#     result = None
#     try:
#         result = Time(meta)
#     except ValueError as err:
#         if "not a valid time string!" not in err.args[0]:
#             raise err
#         else:
#             pass
#     return result
