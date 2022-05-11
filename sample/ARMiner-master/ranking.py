from topic_modeling import model_topics
from helper import *
import numpy as np
from preprocess import *
import os.path
import sys
import shutil

def group_volume(review_group_matrix):
	return (review_group_matrix.sum(axis=0))

def group_average_rating(review_group_matrix, volumes, ratings):
	group_count = volumes.shape[0]
	average_ratings = np.array([(volumes[g] / np.dot(review_group_matrix[:, g], ratings)) for g in range(group_count)])
	return average_ratings

def get_group_rankings(review_group_matrix, ratings,group_weights):
    volumes = group_volume(review_group_matrix)
    high_volume = max(volumes)
    average_ratings = group_average_rating(review_group_matrix, volumes, ratings)
    group_count = volumes.shape[0]
    group_scores = [ (group_weights[0] * float(volumes[idx]/high_volume) + group_weights[1]*average_ratings[idx],idx+1) for idx in range(group_count)]
    group_scores.sort(reverse=True)
    #print("Group scores: ", group_scores)
    scores = np.zeros(group_count)
    rankings = np.zeros(group_count)
    for i, (x, y) in enumerate(group_scores):
        scores[y - 1] = x
        rankings[y - 1] = i + 1
    #print("Scores: ", scores)
    #print("Rankings: ", rankings)
    return (rankings, scores)

def review_ratings(informative_reviews):
	ratings = np.zeros(informative_reviews.shape[0])
	for idx, instance in enumerate(informative_reviews) :
		ratings[idx] = get_rating(instance)
		if ratings[idx] == -1 :
			return ("No rating allocated")
	return ratings

def get_group_weights(groups):
	return np.full(groups, 1/groups)

def get_instance_weights(instances):
	return np.full(instances, 1/instances)

def jaccard_sim(instance1, instance2):
	count_intersect = 0
	count_1 = len(instance1)
	count_2 = len(instance2)
	i = 0
	j = 0
	while(i<count_1 and j <count_2):
		if instance1[i]==instance2[j]:
			count_intersect = count_intersect+1
			i = i+1
			j = j+1
		elif instance1[i]>instance2[j]:
			j = j+1
		else :
			i = i+1
	return float(count_intersect/(count_1 + count_2 - count_intersect))

def instance_ranking(useful_data, review_group_matrix, pred_prob, group_number,rank_number,reverse_mapping):

    is_duplicate = [0 for i in range(len(useful_data))]
    unique_useful_data = []
    unique_data_count = []
    unique_data_proportion = []
    unique_pred_prob = []

    attr_list=[]

    for i in range(len(useful_data)):
    	tmp = useful_data[i]
    	attr_temp=[]
    	for j in range(len(tmp)):
    		if tmp[j]: attr_temp.append(j)
    	attr_list.append(attr_temp)

    similarity_cutoff = 0.8

    attr_tuple = [tuple(x) for x in attr_list]

    for i in range(0,len(attr_list)):

        if is_duplicate[i] != 0:
        	continue

        data_count = 1
        tmp = useful_data[i]

        rating = get_rating(tmp)
        prop = review_group_matrix[i][group_number]
        pred_prob_instance = pred_prob[i]

        for j in range(i,len(attr_list)):
            if jaccard_sim(attr_list[i],attr_list[j]) >= similarity_cutoff:
                is_duplicate[j] = 1
                data_count += 1
                prop = max(prop, review_group_matrix[j][group_number])
                pred_prob_instance = max(pred_prob_instance, pred_prob[j])

                if get_rating(useful_data[j]) < rating:
                	for r in range(6):
                		tmp[r] = 0
                	tmp[get_rating(useful_data[j])] = 1

        unique_data_proportion.append(prop)
        unique_useful_data.append(tmp)
        unique_data_count.append(data_count)
        unique_pred_prob.append(pred_prob_instance)

    proportion = []
    duplicates = []
    rating = []

    for i in range(len(unique_useful_data)):
    	proportion.append(unique_data_proportion[i])
    	duplicates.append(unique_data_count[i])
    	temp =1
    	for x in range(6):
    		if unique_useful_data[i][x]: temp=max(temp,x)
    	rating.append(float(1/temp))

    max_duplicates = max(duplicates)

    instance_weights = [float(1/2), float(1/10), float(1/10), float(3/10)]
    instance_scores =[(instance_weights[0]*proportion[i]+instance_weights[1]*float(duplicates[i]/max_duplicates)+instance_weights[2]*rating[i]+instance_weights[3]*unique_pred_prob[i],i+1) for i in range(len(unique_useful_data))]
    instance_scores.sort(reverse=True)
    instance_rankings =[instance_scores[i][1] for i in range(len(instance_scores))]
    ranked_useful_data = [useful_data[x-1] for x in instance_rankings]

    subdirectory = "groups"
    try:
        os.mkdir(subdirectory)
    except FileExistsError:
        pass

    get_all_instance_words(reverse_mapping,
                           ranked_useful_data,
                           "groups/group_"+str(group_number)+"_rank_"+str(rank_number)+".txt")

    return (unique_useful_data, ranked_useful_data)

def create_groups(useful_data, review_group_matrix, pred_prob):
    num_of_groups = len(review_group_matrix[0])

    groups = [[] for i in range(num_of_groups)]
    review_group_matrix_groups = [[] for i in range(num_of_groups)]
    pred_prob_groups = [[] for i in range(num_of_groups)]

    group_cutoff = 0.3

    for i in range(0,len(useful_data)):
        j = 0
        for j in range(0,num_of_groups):
            if review_group_matrix[i][j] >= group_cutoff:
                groups[j].append(useful_data[i])
                review_group_matrix_groups[j].append(review_group_matrix[i])
                pred_prob_groups[j].append(pred_prob[i])

    # for i in range(num_of_groups):
    #     print("Len Pred Prob: ",len(pred_prob_groups[i]))
    #     print("Len group: ",len(groups[i]))

    return groups, review_group_matrix_groups, pred_prob_groups

def main(app_name):

    # Remove groups subfolder
    try:
        shutil.rmtree("groups")
    except FileNotFoundError:
        pass

    print("Data set: %s\n" % (app_name))
    training_data_list = ["datasets/" + app_name + "/trainL/info.txt",
                            "datasets/" + app_name + "/trainL/non-info.txt"]
    training_data_info = "datasets/" + app_name + "/trainL/info.txt"
    training_data_noninfo = "datasets/" + app_name + "/trainL/non-info.txt"
    test_data = "datasets/" + app_name + "/trainU/unlabeled.txt"

    review_group_matrix, useful_data, mapping, pred_prob = model_topics(training_data_list,
    														 training_data_info,
    														 training_data_noninfo,
    														 test_data)

    ratings = review_ratings (useful_data)
    reverse_mapping = get_reverse_mapping(mapping)

    instance_weights = get_instance_weights(review_group_matrix.shape[0])
    group_weights = get_group_weights(2)
    group_rankings, group_scores = get_group_rankings(review_group_matrix, ratings, group_weights)

    groups, review_group_matrix_groups, pred_prob_groups = create_groups(useful_data, review_group_matrix, pred_prob)

    group_ranking_result = []
    instance_ranking_result = []
    rank = 1
    for idx in range(len(group_rankings)):
        group_rank, instance_rank = instance_ranking(groups[idx], review_group_matrix_groups[idx], pred_prob_groups[idx], idx, rank,reverse_mapping)
        group_ranking_result.append(group_rank)
        instance_ranking_result.append(instance_rank)
        rank = rank + 1
        print("Finished ranking group #%d" % (idx + 1))

    return (group_ranking_result, instance_ranking_result, group_rankings, group_scores, mapping)

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ["facebook", "swiftkey", "tapfish", "templerun2"]:
        main("tapfish")
    else:
        main(sys.argv[1])
