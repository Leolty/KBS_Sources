from topic_modelling_EM_NB import model_topics
from helper import get_rating
import numpy as np
import preprocess


def group_volume(review_group_matrix):
	return (review_group_matrix.sum(axis=0))

def group_average_rating(review_group_matrix, volumes, ratings):
	group_count = volumes.shape[0]
	average_ratings = np.array([(volumes[g] / np.dot(review_group_matrix[:, g], ratings)) for g in range(group_count)])	
	return average_ratings

def get_group_rankings(review_group_matrix, ratings,group_weights):
	volumes = group_volume(review_group_matrix)
	print(volumes)
	average_ratings = group_average_rating(review_group_matrix, volumes, ratings)
	group_count = volumes.shape[0]
	group_scores = [ (group_weights[0] * volumes[idx] + group_weights[1]*average_ratings[idx],idx+1) for idx in range(group_count)]
	group_scores.sort()
	rankings = [ group_scores[idx][1] for idx in range(group_count)]
	return rankings

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

def main():

	training_data_list = ["datasets/swiftkey/trainL/info.txt","datasets/swiftkey/trainL/non-info.txt"]
	training_data_info = "datasets/swiftkey/trainL/info.txt"
	training_data_noninfo = "datasets/swiftkey/trainL/non-info.txt"
	test_data_info = "datasets/swiftkey/test/info.txt"
	test_data_noninfo = "datasets/swiftkey/test/non-info.txt"
	trainU_data = "datasets/swiftkey/trainU/unlabeled.txt"

	review_group_matrix, useful_data, mapping = model_topics(training_data_list, 
															 training_data_info, 
															 training_data_noninfo, 
															 test_data_info,
															 test_data_noninfo,
															 trainU_data)

	ratings = review_ratings (useful_data)
	
	instance_weights = get_instance_weights(review_group_matrix.shape[0])
	group_weights = get_group_weights(2)
	group_rankings = get_group_rankings(review_group_matrix, ratings, group_weights)
	print(group_rankings)

if __name__ == '__main__':
	main()
