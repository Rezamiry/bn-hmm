import pickle
import pandas as pd
import numpy as np

from hmm_bnetwork import HMMBNetwork
from nb_hmm import NaiveBayesHMM


print("Loading dataset")
levels = 2
with open("train_test_pca_5_level_{}.pickle".format(levels), "rb") as f:
    dataset = pickle.load(f)


train_set = dataset["train_set"]
test_set = dataset["test_set"] 


class_name = "state"
class_states_count = 2
nodes_info = {}
for column in train_set['1'][0].columns:
    nodes_info[str(column)] = levels

root_node = train_set['1'][0].columns[0]

print("Initializing training")

models_dic = {}
for key, value in train_set.items():
    print("training {} model".format(key))
    model_train = HMMBNetwork.initialize(class_name, class_states_count, nodes_info, root_node)
    df_list = []
    for df in train_set[key]:
        df.columns = [str(col) for col in df.columns]
        df_list.append(df)
    model_train.train_multiple_observations(df_list, epoches=10, verbose=1)
    models_dic[key] = model_train

with open(f"hmm_bnetwork_motion_trained_models_pca_5_level_{levels}.pickle", "wb") as f:
    pickle.dump(models_dic, f)
