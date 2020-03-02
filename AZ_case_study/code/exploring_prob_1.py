import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.query_utils import get_list_ordered_by_counts, get_count_matrix
from utils.distances import compute_distance_matrix
from utils.data_prep import prepare_data_for_matrix_viz

# --------------------------------------------------------------------------
# Exploration 1: Are different adverse events reported in different countries?
# --------------------------------------------------------------------------

# Step 1: Finding the list of unique medical reactions (top 100 reactions)
# ------------------------------------------------------------------------
adv_event_list, _ = get_list_ordered_by_counts(count_field='patient.reaction.reactionmeddrapt.exact', limit=100)


# Step 2: For each reaction in this list (from step 1) find the unique count of occurrence countries
# ---------------------------------------------------------------------------------------------------

count_matrix_events_countries = get_count_matrix(row_field='patient.reaction.reactionmeddrapt.exact',
                                                 column_field='occurcountry.exact',
                                                 row_limit=100, column_limit=1000, fill_na=0)

# Step 3: Normalizing the raw counts by the number of events reported from the respective countries
# (this will give us a likelihood score for each reaction on a country level)
# -------------------------------------------------------------------------------------------------

norm_matrix_df = prepare_data_for_matrix_viz(count_matrix_events_countries, normalize=True,
                                             transpose_data=True, norm_axis=0)

# Step 4: Visualizing results using a heat-map
# --------------------------------------------

plot = sns.heatmap(np.power(norm_matrix_df, 0.2))
plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.xlabel('Adverse event')
plt.ylabel('Country of occurrence')
plt.title('Scaled probability of event occurrence for each country\n(power scaling of 0.2 used for clarity)')

# Step 5: Visualizing country clusters (using T-SNE) by treating
# the response normalized vectors as country wise features
# ---------------------------------------------------------------

# Step 5.1: Constructing the distance matrix
# (using KL Divergence as our distance metric since we are comparing probability distributions)

dist_matrix = compute_distance_matrix(norm_matrix_df, metric='kl_divergence')

# Step 5.2: Visualize clusters in 2 dimensions
np.random.seed(4)  # Setting random seed for consistency
cluster_data = TSNE(n_components=2, metric='precomputed').fit_transform(dist_matrix)

# Plot cluster data points in 2D
fig, ax = plt.subplots()
ax.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5, c='lightcoral')
for idx in range(len(dist_matrix)):
    ax.annotate(norm_matrix_df.index[idx], (cluster_data[idx, 0], cluster_data[idx, 1]), fontsize=7)
plt.title('Visualizing the difference in response between different countries\n(using t-SNE)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# -------------------------------------------------------------------------
# Conclusion:
# -----------
# By analyzing the heatmap and t-SNE plots, we can conclude that there is
# indeed an observable difference in the type of adverse events reported
# across different countries (these could be due to many factors - geographic,
# economic, etc.)

# Insights:
# ---------

# 1. Cases of death are most probably reported in every country but
# events like 'hospitalization' or 'visual impairment' are not / seldom reported in
# many countries. This might be because these adverse events are rare by
# nature or the non-reporting countries lack clear guidance in reporting
# such events.

# 2. Countries like Philippines, China and South Korea report many adverse event types but
# countries like Laos and Maldives report very few event types. This is most
# likely because the number of event reports coming from the later countries
# could be very few compared to other countries (shown in the following graph).
# Maldives has just 5 event reports filed so far! (from Jan 1st 2004 to Dec 31st 2019)

# Country-wise number of event reports plot (paired with population per country as of 2019)
# -----------------------------------------------------------------------------------------

country_list, event_report_count = get_list_ordered_by_counts(count_field='occurcountry.exact', limit=1000)
population_list = [1.4*10**9, 108.12*10**6, 51.23*10**6, 10.16*10**6, 7.17*10**6, 0.39*10**6]  # From Google
countries_of_interest = ['CN', 'PH', 'KR', 'LA', 'MV']

country_count_dict = dict(zip(country_list, event_report_count))
country_population_dict = dict(zip(countries_of_interest, population_list))
bar_width = 0.25

_, ax = plt.subplots()
bar_1 = ax.bar(np.arange(5), np.log10([country_count_dict[key] for key in countries_of_interest]), bar_width)
bar_2 = ax.bar(np.arange(5) + bar_width, np.log10([country_population_dict[key] for key in countries_of_interest]), bar_width)
ax.set_xticks(np.arange(5) + bar_width / 2)
ax.set_xticklabels(countries_of_interest)
ax.legend((bar_1[0], bar_2[0]), ('No. of event reports', 'Population (2019)'))

plt.xlabel('Country code')
plt.ylabel('Logarithm (base 10) of number of reports / population')
plt.title('Number of reports filed for events that occurred in each country')


# Note:
# ------

# This analysis can be extend to each medicine or disease type by making
# simple adjustments to the search queries
# -------------------------------------------------------------------------


# -----------------------------------------END-OF-CODE-----------------------------------------------------------------














