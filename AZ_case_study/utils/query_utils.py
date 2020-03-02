import requests
import pandas as pd
from collections import OrderedDict
from config import date_string, api_key


def get_list_ordered_by_counts(count_field='patient.reaction.reactionmeddrapt.exact', limit=100, filter_to_use=None, custom_date_range=None):

    """
    Queries the top 'N' values according to their frequency of occurrence through the openFDA API
    ------------------------------------------------------------------------------------------------------

    :param limit: (int) Number of entries to return from the API (values higher than 100 might slow down the code)
    :param count_field: (str) Field identifier string for which unique counts are required
    :param filter_to_use: (str) Additional filter to be applied before deriving the counts
    :param custom_date_range: (str) Custom date range based on which the query should pull the data
                                   (format: [star_date+TO+end_date])

    :return term_list: (list) List of top entries matching the query (arranged in the descending order of count)
    :return count_list: (list) Term counts for entries in 'term_list'
    """

    params = OrderedDict()
    params['api_key'] = api_key
    params['count'] = count_field
    params['limit'] = limit

    date_string_to_use = date_string if custom_date_range is None else custom_date_range

    # Correcting filter for special characters
    # (based on errors observed in Crohn's and Parkinson's disease indication)
    filter_string_to_use = date_string_to_use if filter_to_use is None else date_string_to_use + '+AND+' + filter_to_use.replace("^", "'")

    count_data = requests.get('https://api.fda.gov/drug/event.json?search=receivedate:' + filter_string_to_use,
                              params=params)

    term_list = []
    count_list = []
    try:
        for results in count_data.json()['results']:
            term_list.append(results['term'])
            count_list.append(results['count'])
    except KeyError:
        # print('Error status code : ', count_data.status_code)
        # print('URL request : ', count_data.url)
        pass
    
    return term_list, count_list


def get_count_matrix(row_field, column_field, row_limit=100, column_limit=100, fill_na=0, custom_date_range=None, filter_to_use=None):

    """
    Constructs a count matrix according to the row and column variables specified by the user
    ------------------------------------------------------------------------------------------

    :param row_field: (str) Field to be used to construct the rows names (dimension=0)
    :param column_field: (str) Field to be used to construct the columns names (dimension=1)
    :param row_limit: (int) Number of top row names (by count) to be queried from the API
    :param column_limit: (int) Number of top column names (by count) to be queried from the API
    :param fill_na: (numeric) Value used for filling NaNs
    :param custom_date_range: (str) Custom date range based on which the query should pull the data
                                   (format: [star_date+TO+end_date])
    :param filter_to_use: (str) Additional filter to be applied before deriving the counts


    :return: (pandas DataFrame) A DataFrame representing the count matrix with rows and columns indexed
            (The number of columns might be higher than 'column_limit' because each row entry might have unique
            top column elements when queried and some of them would not have been covered by the previous row elements)
    """

    # Get the name list for the row dimension
    row_name_list, _ = get_list_ordered_by_counts(count_field=row_field, limit=row_limit, filter_to_use=filter_to_use, custom_date_range=custom_date_range)

    query_df = pd.DataFrame()  # Empty DataFrame to which data is added row-wise
    for row_idx, row_name in enumerate(row_name_list):
        # Get the unique count of column variable for each row variable under consideration
        row_filter_text = row_field + ':' + '"' + str(row_name) + '"'

        # Note: Adding additional filters might restrict the search too much and no records might be found
        # In such cases, the error code is printed and the respective rows are filled with zero values
        filter_text = row_filter_text + '+AND+' + filter_to_use if filter_to_use is not None else row_filter_text

        column_name_list, column_count_list = get_list_ordered_by_counts(count_field=column_field, limit=column_limit,
                                                                         filter_to_use=filter_text, custom_date_range=custom_date_range)

        col_df = pd.DataFrame(columns=column_name_list)
        col_df.loc[row_idx, :] = column_count_list
        query_df = pd.concat([query_df, col_df], sort=True, axis=0)

    query_df.fillna(fill_na, inplace=True)
    query_df.index = row_name_list

    return query_df







