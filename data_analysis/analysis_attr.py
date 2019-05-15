import matplotlib.pyplot as plt 
import pandas 

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


# every row has 5 attributes.
all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
            'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
            'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
]


pos, neg = stat_attr("../CelebA/Anno/list_attr_celeba.txt")
attr_dict = {}
with open('sample_num.csv', 'w') as f:
    f.write("attribute,positive sample,negative sample\n")
    for i, attr in enumerate(all_attrs):
        f.write(attr + ',' + str(pos[i]) + "," + str(neg[i]) + '\n')
