"""
Author: Yihao Qiu
Date: 2019-03-16
Description: Read the attributes and analysis the distribution of Celeba dataset.
"""
import matplotlib.pyplot as plt 

attr_name = [""]

# calculate the number of positive and negetive samples of each attribute
def stat_attr(file_path):

    pos_samples = [0 for i in range(40)]
    neg_samples = [0 for i in range(40)]
    #attri_id = [i + 1 for i in range(40)]
    #attr_name = []
    
    with open(file_path) as f:
        attr_info = f.readlines()
        #attr_name = attr_info[1].split()
        attr_id = [i + 1 for i in range(40)]
        attr_info = attr_info[2:]
        index = 0
        
        for line in attr_info:
            index += 1
            sample_info = line.split()
            for i in range(len(sample_info)):
                if i != 0:
                    if sample_info[i] == '1':
                        pos_samples[i - 1] = pos_samples[i - 1] + 1
                        
                    elif sample_info[i] == '-1':
                        neg_samples[i - 1] = neg_samples[i - 1] + 1
    fig, ax = plt.subplots()                
    ax.plot(attr_id, pos_samples, color = 'r', marker="o", label = "Positive Samples")
    ax.plot(attr_id, neg_samples, color = 'g', marker=".", label = "Negetive Samples")
    plt.title("Show the negetive and positive attributes of Celeba")
    plt.xlabel("The attribute")
    plt.ylabel("The number of samples")
    ax.legend()
    plt.show()
    return pos_samples, neg_samples


stat_attr("./Anno/list_attr_celeba.txt")
