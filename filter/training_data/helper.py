import numpy as np

# this function writes all the instances and the words corresponding to a instance
def get_all_instance_words(reverse_mapping,data,file_name):

    # word id starts from zero
    rating_bits = 6
    instance_count = 1

    word_list = open(file_name,"w")

    # prints the instances and the words per instance for reference
    for line in data:
        word_id = 0
        # excluded the rating attributes
        line = line[rating_bits:]
        meta="------Rating Instance:"+str(instance_count)+"--------\n"
        word_list.write(meta)
        for word_id in range(len(line)):
        # checks if the value corresponding to that attribute is non-zero
            if line[word_id] != 0:
                word = reverse_mapping.get(word_id, None)
                if word is not None:
                    word_list.write(word + "\n")
        instance_count = instance_count + 1

    word_list.close()

# this function prints all the words corresponding to an instance
def get_instance_words(reverse_mapping,data):
    # word id starts from zero
    rating_bits = 6
    instance_count = 1

    # get all the non-zero indices
    indices = np.nonzero(data)[0][1:]
    return_val = [reverse_mapping[index - rating_bits] for index in indices]
    for word in return_val:
    	print(word)
    return return_val

def get_rating(instance):
    for i in range(6):
        if instance[i] == 1 :
            return i
    return -1
