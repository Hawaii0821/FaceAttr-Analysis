"""


"""
import numpy as numpy
import matplotlib.pyplot as plt 
import seaborn as sns 

"""draw a scatter plot for "precision" and "recall"""
def drawPRCurve(model_list, p_list, r_list, save_path): 
    fig, ax = plt.subplots()
    for i, model_name in enumerate(model_list):
        ax.scatter(r_list[i], p_list[i],s=100, label=model_name, alpha=0.9, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.xlabel("recall/%")
    plt.ylabel("precision/%")
    plt.title("The precision and recall scatter")
    plt.xlim([63, 67])
    plt.ylim([79, 80])
    plt.savefig(save_path)
    plt.show()

""" draw histgram """
def drawBarPlot(model_list, data_list, y_label, save_path, title, label="accuracy"):
    assert label == "accuracy" or label == "f1"
    fix, ax = plt.subplots()
    sns.barplot(model_list, data_list, palette="rocket", ax=ax)
    # ax.axhline(91.5, color="k", clip_on=True)
    if label=="accuracy":
        plt.ylim([91, 92])
    elif label=="f1":
        plt.ylim([68, 71])
    ax.set_ylabel(y_label)
    plt.xticks(rotation = 10)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":

    data = {
        "Resnet101": {"accuracy": 91.53, "precision": 79.81, "recall": 63.67, "F1": 68.52},
        "GCnet101": {"accuracy": 91.94, "precision": 79.45, "recall": 65.64, "F1": 68.94},
        "SEnet101": {"accuracy": 91.95, "precision": 79.81, "recall": 66.64, "F1": 69.99},
        "SGEnet101": {"accuracy": 91.60, "precision": 79.23, "recall": 65.40, "F1": 69.77},
        "SKnet101": {"accuracy": 91.93, "precision": 79.69, "recall": 65.54, "F1": 69.95},
        "CBAMnet101": {"accuracy": 91.42, "precision": 79.12, "recall": 65.32, "F1": 69.24},
        "Densenet121": {"accuracy": 91.64, "precision": 79.25, "recall": 65.42, "F1": 69.81}
    }

    model_list = []
    acc_list = []
    p_list = []
    r_list = []
    f_list = []

    for key, value in data.items():
        model_list.append(key);
        print(key)
        print(value)
        for k, v in value.items():
            if k == "accuracy":
                acc_list.append(v)
            if k == "precision":
                p_list.append(v)
            if k == "recall":
                r_list.append(v)
            if k == "F1":
                f_list.append(v)

    drawPRCurve(model_list, p_list, r_list, "pr_scatter");
    drawBarPlot(model_list, acc_list, "accuracy/%",  "acc_curve","The accuracy of models", "accuracy")
    drawBarPlot(model_list, f_list,"F1/%", "f1_curve","The F1 of models", "f1")
    