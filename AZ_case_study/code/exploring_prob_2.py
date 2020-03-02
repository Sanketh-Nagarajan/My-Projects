import numpy as np
import matplotlib.pyplot as plt
from utils.query_utils import get_list_ordered_by_counts, get_count_matrix

# ----------------------------------------------------------------------------------------------
# Exploration 2: What are the different adverse events associated with different disease areas?
# ----------------------------------------------------------------------------------------------

# Get top 100 disease areas (by count)
disease_area_list, _ = get_list_ordered_by_counts(count_field='patient.drug.drugindication.exact', limit=100)
# Observation: The disease indication is unknown for most cases

# Find unique count of each adverse event according to disease area
disease_vs_event_count_matrix = get_count_matrix(row_field='patient.drug.drugindication.exact',
                                                 column_field='patient.reaction.reactionmeddrapt.exact',
                                                 row_limit=100, column_limit=100, fill_na=0)

# Dropping duplicate rows since Crohn's disease has 2 variants
disease_vs_event_count_matrix = disease_vs_event_count_matrix.drop_duplicates()

# Observation: The adverse event field sometimes also contains the reasons for these reactions
# (sample list shown below)
print(disease_vs_event_count_matrix.columns[-5:])

# Finding TF-IDF weights for each adverse event
# (this statistic measures the extent of association an adverse event has with a particular disease area)
# -------------------------------------------------------------------------------------------------------

# Step 1: Computing term-frequency for each adverse event across disease areas
# (In this case, each adverse event is a term and each disease area is a document type)
tf_matrix = disease_vs_event_count_matrix / disease_vs_event_count_matrix.sum(axis=1).values.reshape(-1, 1)

# Step 2: Computing the inverse document frequency (IDF)
# IDF = Log10(Total number of records / Number of records with the specific adverse event contained in it)
idf_vector = np.log10(np.sum(disease_vs_event_count_matrix.values) / disease_vs_event_count_matrix.sum(axis=0))

# Step 3: Construct TF-IDF weighted matrix
# This says when an adverse event rarely occurs (across event reports) but occurs frequently under one particular
# disease area, then it is most likely associated with that disease area
tf_idf_matrix = tf_matrix * idf_vector.values.reshape(1, -1)


# Visualizing associated adverse events for a given disease area
# ---------------------------------------------------------------

def find_associated_adverse_events(disease_ind, top_num=10):
    """
    # Finding the top 'N' associated adverse events for a given disease area
    -------------------------------------------------------------------------

    :param disease_ind: (str) Disease indicator for which the 'top_num' associated events are to be displayed
    :param top_num: (int) Number of top associated events to display

    :return: (pandas Series) Top 'N' associated adverse events with index as event names
    """
    associated_events = tf_idf_matrix.loc[tf_idf_matrix.index == disease_ind].T.\
                            sort_values(by=[disease_ind], ascending=False)[:top_num]

    # Plotting results (sorted in the order of significance)
    _, ax = plt.subplots()
    ax.bar(np.arange(len(associated_events)), associated_events.values.flatten(), 0.35)
    ax.set_xticks(np.arange(len(associated_events)))
    ax.set_xticklabels(['\n'.join(ind.split(' ')) for ind in associated_events.index.values], rotation=30,
                       horizontalalignment='right')
    plt.xlabel('Associated adverse events')
    plt.ylabel('Degree of association to - ' + disease_ind)
    plt.title('Adverse events associated with : ' + disease_ind)
    plt.show()

    return associated_events


# Finding top 10 associated adverse events for top 6 disease areas
num_top_disease_areas_to_plot = 6
for disease_area in disease_area_list[1:num_top_disease_areas_to_plot + 1]:
    _ = find_associated_adverse_events(disease_area, 10)

# ---------------------------------------------------------------------------------------------------------------------
# Conclusion
# ----------
# Different disease areas tend to have a set of associated adverse events, but the most likely ones are related to
# disease relapses or ineffectiveness of the drug.

# Observations
# ------------
# 1. The disease indication is unknown for most cases
# 2. The adverse event field sometimes also contains the reasons for these reactions.
# For example: 'WRONG DRUG ADMINISTERED', 'WRONG PATIENT RECEIVED MEDICATION',
    # 'WRONG TECHNIQUE IN DEVICE USAGE PROCESS', 'WRONG TECHNIQUE IN DRUG USAGE PROCESS'

# Note:
# -----
# The associations should NOT be interpreted as that the occurrence of adverse event 'A' is most likely when patients
# take medications for disease area 'D' (as a causal relationship).
# Serious adverse events are usually given higher priority when it comes to reporting.
# Thus, in case of diabetes mellitus for example, 'myocardial infarction' might not be the most likely side effect of
# diabetes related drugs. But diabetic medications involving such adverse events are most likely to be reported.
# ---------------------------------------------------------------------------------------------------------------------


# -----------------------------------------END-OF-CODE-----------------------------------------------------------------





