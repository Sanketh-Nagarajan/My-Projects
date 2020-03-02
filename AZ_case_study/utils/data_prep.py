import numpy as np


def prepare_data_for_matrix_viz(data, normalize=True, transpose_data=True, norm_axis=0):
    """
    Normalize and sort the raw count data for heat-map visualizations
    ------------------------------------------------------------------

    :param data: (pandas DataFrame) DataFrame containing data to be normalized
    :param normalize: (Boolean (set to True by default)) If True, then data is normalized along norm_axis
    :param transpose_data: (Boolean (set to True by default)) If True, then rows and columns of the data are swapped
                          (a matrix transpose)
    :param norm_axis: (int) Axis along which the data need to be normalized

    :return: (pandas DataFrame) Normalized, transposed and sorted data
    """

    # Normalizing the data
    if normalize:
        if norm_axis == 1:
            # Row-wise normalization
            data_norm = data / data.sum(axis=norm_axis).values.reshape(-1, 1)
        else:
            # Column-wise normalization
            data_norm = data / data.sum(axis=norm_axis).values.reshape(1, -1)
    else:
        data_norm = data.copy()

    # Sorting by row and column totals (data presence wise) for visual appeal
    if transpose_data:
        data_norm = data_norm.T

    data_norm['row_wise_count'] = data_norm.apply(lambda row: sum(row > 0), axis=1)
    data_norm.loc['column_wise_count', :] = data_norm.apply(lambda col: sum(col > 0), axis=0)
    data_norm = data_norm.sort_values(by=['row_wise_count'], ascending=False, axis=0)
    data_norm = data_norm.sort_values(by=['column_wise_count'], ascending=False, axis=1)
    data_norm.drop(columns=['row_wise_count'], inplace=True)
    data_norm.drop(['column_wise_count'], axis=0, inplace=True)

    return data_norm





