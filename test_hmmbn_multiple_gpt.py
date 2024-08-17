import csv
import random
import pandas as pd
import numpy as np

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from hmmtan import HMMTAN
from nb_hmm import NaiveBayesHMM
from hmm_bnetwork import HMMBNetwork

# Configuration parameters
class_name = "A"
class_states_count = 2
root_node = "B"
nodes_info = {"B": 4, "C": 4, "D": 4, "E": 4, "F": 4}


def convert_observations_to_nbhmm(o_history):
    return [[o[key] for o in o_history] for key in sorted(o_history[0].keys())]


def initialize_models():
    return [HMMBNetwork.initialize(class_name, class_states_count, nodes_info) for _ in range(6)]


def simulate_training_data(models, num_training_instances, train_instance_size):
    tan_train_list = [[] for _ in range(6)]
    nb_train_list = [[] for _ in range(6)]
    for model_index, model in enumerate(models):
        for _ in range(num_training_instances):
            _, o_history = model.simulate(train_instance_size)
            train_df = pd.DataFrame.from_dict(o_history)
            observations_nb = convert_observations_to_nbhmm(o_history)
            tan_train_list[model_index].append(train_df)
            nb_train_list[model_index].append(observations_nb)
    return tan_train_list, nb_train_list


def train_models(tan_train_list, nb_train_list, epoches):
    tan_models = [HMMTAN.initialize(class_name, class_states_count, nodes_info, root_node) for _ in range(6)]
    bnetwork_models = [HMMBNetwork.initialize(class_name, class_states_count, nodes_info) for _ in range(6)]
    states = list(range(class_states_count))
    observables = [list(range(count)) for count in nodes_info.values()]
    nb_models = [NaiveBayesHMM.initialize(states, observables) for _ in range(6)]
    
    for i in range(6):
        tan_models[i].train_multiple_observations(tan_train_list[i], epoches=epoches, verbose=0)
        bnetwork_models[i].train_multiple_observations(tan_train_list[i], epoches=epoches, verbose=0)
        nb_models[i].train_multiple_observations(nb_train_list[i], epoches)
    
    return tan_models, bnetwork_models, nb_models


def evaluate_models(models, tan_models, bnetwork_models, nb_models, num_test_instances, test_instance_size):
    results = []
    for i in range(num_test_instances):
        for model_index, model in enumerate(models):
            _, o_history = model.simulate(test_instance_size)
            test_df = pd.DataFrame.from_dict(o_history)
            test_nb = convert_observations_to_nbhmm(o_history)
            
            result = {"label": model_index + 1}
            for j in range(6):
                result[f"hmmtan_{j + 1}"] = tan_models[j].score(test_df)
                result[f"hmmnb_{j + 1}"] = nb_models[j].score(test_nb)
                result[f"hmmbnet_{j + 1}"] = bnetwork_models[j].score(test_df)
            
            results.append(result)
    return results


def calculate_accuracy(results_df):
    tan_cols = [col for col in results_df.columns if col.startswith("hmmtan")]
    results_df['pred_tan'] = results_df[tan_cols].idxmax(axis=1).str.replace("hmmtan_", "").astype(int)

    nb_cols = [col for col in results_df.columns if col.startswith("hmmnb")]
    results_df['pred_nb'] = results_df[nb_cols].idxmax(axis=1).str.replace("hmmnb_", "").astype(int)

    bnet_cols = [col for col in results_df.columns if col.startswith("hmmbnet")]
    results_df['pred_bnet'] = results_df[bnet_cols].idxmax(axis=1).str.replace("hmmbnet_", "").astype(int)

    tan_accuracy = (results_df['pred_tan'] == results_df['label']).mean()
    nb_accuracy = (results_df['pred_nb'] == results_df['label']).mean()
    bnet_accuracy = (results_df['pred_bnet'] == results_df['label']).mean()

    return tan_accuracy, nb_accuracy, bnet_accuracy

def test(config, output_file):
    num_tests = config['num_tests']
    num_training_instances = config['num_training_instances']
    train_instance_size = config['train_instance_size']
    num_test_instances = config['num_test_instances']
    test_instance_size = config['test_instance_size']
    epoches = config['epoches']

    test_results = []
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['test_number', 'hmmbnet_accuracy', 'hmmtan_accuracy', 'hmmnb_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for test_number in range(num_tests):
            models = initialize_models()
            tan_train_list, nb_train_list = simulate_training_data(models, num_training_instances, train_instance_size)
            tan_models, bnetwork_models, nb_models = train_models(tan_train_list, nb_train_list, epoches)

            results = evaluate_models(models, tan_models, bnetwork_models, nb_models, num_test_instances, test_instance_size)
            results_df = pd.DataFrame.from_dict(results)

            tan_accuracy, nb_accuracy, bnet_accuracy = calculate_accuracy(results_df)

            print(f"Test {test_number} - hmmbnet accuracy = {bnet_accuracy * 100:.2f}% | hmmtan accuracy = {tan_accuracy * 100:.2f}% | hmmnb accuracy = {nb_accuracy * 100:.2f}%")
            test_results.append({"test_number": test_number, "hmmbnet_accuracy": bnet_accuracy, "hmmtan_accuracy": tan_accuracy, "hmmnb_accuracy": nb_accuracy})
            
            # Write intermediate results to CSV
            writer.writerow({"test_number": test_number, "hmmbnet_accuracy": bnet_accuracy, "hmmtan_accuracy": tan_accuracy, "hmmnb_accuracy": nb_accuracy})
    
    return test_results


configs = [{
    "num_tests": 50,
    "epoches": 25,
    "train_instance_size": 20,
    "num_training_instances": 10,
    "num_test_instances": 50,
    "test_instance_size": 20}]


# }, {
#     "num_tests": 100,
#     "epoches": 30,
#     "train_instance_size": 50,
#     "num_training_instances": 10,
#     "num_test_instances": 100,
#     "test_instance_size": 10,
# }, {
#     "num_tests": 100,
#     "epoches": 30,
#     "train_instance_size": 100,
#     "num_training_instances": 10,
#     "num_test_instances": 100,
#     "test_instance_size": 10,
# }]

test_results = []
for test_index, config in enumerate(configs):
    print(f"Test {test_index} - with config {config}")
    output_file = f"results/test_hmmbn_6_class_long_run_test_config_{test_index}.csv"
    test_result = test(config, output_file)
    test_results.append({"test_id": test_index, "config": config, "test_result": test_result})

import pickle
with open("results/hmm_bnet_multiple_training_test_true_bn_longer.pickle", "wb") as f:
    pickle.dump(test_results, f)
