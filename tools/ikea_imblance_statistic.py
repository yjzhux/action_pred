import os, csv, json, ipdb
from collections import Counter
import matplotlib.pyplot as plt

def count_num(root_dir, rec_file):

    num_dict = {}
    name_dict = {}
    rec_path = os.path.join(root_dir, rec_file)
    # annotation list: dataset_public/ikea.../videos/2016-09-01/GOPRO115.MP4.30HZ.json
    with open(rec_path, 'r') as f_csv:
        annot_list = csv.reader(f_csv)

        for row in annot_list:
            # full path of json
            # TO DO: move out of this function
            sub_dirs = ''.join(row).split('/')
            annot_path = os.path.join(root_dir, sub_dirs[-3], sub_dirs[-2], sub_dirs[-1])
            
            with open(annot_path, 'r') as f_json:
                annot = json.load(f_json)
                labels = annot['Y']
                names = annot['Y_labels']
            
            num_dict = dict(Counter(num_dict) + Counter(labels))
            name_dict = dict(Counter(name_dict) + Counter(names))
    
    return num_dict, name_dict

def match_label_name(num_dict, name_dict):
    label_name = {}
    for label, num1 in num_dict.items():
        for name, num2 in name_dict.items():
            if num1 == num2:
                # import ipdb; ipdb.set_trace()
                label_name[label] = name 
                break
    return label_name


def show_counts(labels, numbers, label_name):
    rects = plt.bar(x=labels, height=numbers)
    # import ipdb; ipdb.set_trace()
    # plt.xticks(labels, [label_name[i] for i in labels], rotation=30)
    plt.xticks(labels)
    plt.xlabel('action label')
    plt.ylabel('Number')
    plt.title('Number of frames for each class in mini training dataset')

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), 
                 ha='center', va='bottom', rotation=0, rotation_mode='anchor')
    plt.show()


def loss_weight(numbers):
    s = sum(numbers)
    ratios = [num / s for num in numbers]
    weights = [int(s / num) for num in numbers]
    return weights


if __name__ == '__main__':
    root_dir = '../dataset_public/ikea-fa-release-data/'
    rec_file = 'train_30fps_mini.csv'
    num_dict, name_dict = count_num(root_dir, rec_file)
    label_name = match_label_name(num_dict, name_dict)
    print(label_name)

    labels = []
    numbers = []
    for idx in range(len(num_dict)):
        labels.append(idx)
        numbers.append(num_dict[idx])
    # show_counts(labels, numbers, label_name)
    weights = loss_weight(numbers)
    print(weights)
    # import ipdb; ipdb.set_trace()
    # class_weights = [9, 76, 57, 90, 52, 138, 128, 110, 95, 13, 2, 3, 24]