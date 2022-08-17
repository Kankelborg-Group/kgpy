import matplotlib.cm
import matplotlib.colors

__all__ = [
    'spectral_colors',
]


def spectral_colors() -> matplotlib.cm.ScalarMappable:
    colormap = matplotlib.cm.get_cmap('gist_rainbow')
    segment_data = colormap._segmentdata.copy()

    last_segment = ~1
    segment_data['red'] = segment_data['red'][:last_segment].copy()
    segment_data['green'] = segment_data['green'][:last_segment].copy()
    segment_data['blue'] = segment_data['blue'][:last_segment].copy()
    segment_data['alpha'] = segment_data['alpha'][:last_segment].copy()

    segment_data['red'][:, 0] /= segment_data['red'][~0, 0]
    segment_data['green'][:, 0] /= segment_data['green'][~0, 0]
    segment_data['blue'][:, 0] /= segment_data['blue'][~0, 0]
    segment_data['alpha'][:, 0] /= segment_data['alpha'][~0, 0]

    colormap = matplotlib.colors.LinearSegmentedColormap(
        name='spectrum',
        segmentdata=segment_data,
    )
    mappable = matplotlib.cm.ScalarMappable(
        cmap=colormap.reversed(),
    )
    return mappable
