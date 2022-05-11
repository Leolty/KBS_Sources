from filter_EM_NB import filter
import helper
import preprocess
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation


def group_by_topic(X, mapping, reverse_mapping):

    lat_dir_aloc = LatentDirichletAllocation(n_components=10)
    lat_dir_aloc.fit(X)

# Uncomment if top words in each topic are needed
#    for component in lat_dir_aloc.components_:
#        freq_word_ids = np.argsort(component)[-10:]
#        for word_id in freq_word_ids:
#            word = reverse_mapping.get(word_id, None)
#            if word is not None:
#                print(word, end=', ')
#        print()

    return lat_dir_aloc.transform(X)

def model_topics(training_data_list, training_data_info, training_data_noninfo, test_data_info, test_data_noninfo, trainU_data):

    informative_reviews, mapping, reverse_mapping =  filter(training_data_list, training_data_info, training_data_noninfo, test_data_info, test_data_noninfo, trainU_data)

    useful_data = np.append(informative_reviews, preprocess.get_data([training_data_info], mapping), axis=0)

    group_matrix = group_by_topic(useful_data, mapping, reverse_mapping)
    return (group_matrix, useful_data, mapping)
