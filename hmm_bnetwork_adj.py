import pgmpy
import pandas as pd
import numpy as np
from nb_hmm import ProbabilityMatrix, ProbabilityVector
import random
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import TreeSearch 
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator

class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = probabilities.values()

        assert len(states) == len(probs), \
            "The probabilities must match the states."
        assert len(states) == len(set(states)), \
            "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, \
            "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x:
            probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(state, list(array))))

    @property
    def dict(self):
        return {k: v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError(
                "Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):

        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values = np.stack([prob_vec_dict[x].values
                           for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
             / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array:
                  np.ndarray,
                  states: list,
                  observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x)))
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values,
               columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError(
                "Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)
    

class BNetwork:
    def __init__(self, bn, nodes, class_name, class_states) -> None:
        self.bn = bn
        self.nodes = nodes
        self.class_name = class_name
        self.class_states = class_states
        self.inference = None
    
    @staticmethod
    def create_random_cpd(variable_card_cound, evidence_card_counts=None):
        if evidence_card_counts is None or len(evidence_card_counts) == 0:
            r = np.random.rand(variable_card_cound, 1)
            r = r/r.sum(axis=0)
            return r
        else:
            r_list = []
            for evidence_card_count in evidence_card_counts:
                r = np.random.rand(variable_card_cound, evidence_card_count)
                r = r/r.sum(axis=0)
                r_list.append(r)
            return r_list[0]

    @classmethod
    def initialize(cls, nodes_info, class_name, class_states):
        nodes = list(nodes_info.keys())
        class_state_count = len(class_states)
        random.shuffle(nodes)
        root_node = nodes[0]
        edges = []
        for i in range(len(nodes)-1):
            edges.append((nodes[i], nodes[i+1]))
            edges.append((class_name, nodes[i]))
        edges.append((class_name, nodes[-1]))
        bn = BayesianNetwork(edges)

        cpt_list = []
        cpt_class = TabularCPD(variable=class_name, variable_card=class_state_count, values=cls.create_random_cpd(class_state_count))
        cpt_root_node = TabularCPD(variable=nodes[0], variable_card=nodes_info[nodes[0]], values=cls.create_random_cpd(nodes_info[nodes[0]],[class_state_count]), evidence=[class_name], evidence_card=[class_state_count])
        for i in range(1, len(nodes)): # skip root node
            cpt_node = TabularCPD(variable=nodes[i], variable_card=nodes_info[nodes[i]], values=cls.create_random_cpd(nodes_info[nodes[i]],[nodes_info[nodes[i-1]]*class_state_count]), evidence=[nodes[i-1], class_name], evidence_card=[nodes_info[nodes[i-1]], class_state_count])
            cpt_list.append(cpt_node)
        cpt_list.append(cpt_class)
        cpt_list.append(cpt_root_node)
        
        bn.add_cpds(*cpt_list)


        return cls(bn, nodes, class_name, class_states)
    
    @classmethod
    def initialize_random(cls, nodes_info, class_name, class_states, root_node):
        nodes = list(nodes_info.keys())
        class_state_count = len(class_states)
        edges = []
        for i in range(len(nodes)-1):
            edges.append((nodes[i], nodes[i+1]))
            edges.append((class_name, nodes[i]))
        edges.append((class_name, nodes[-1]))
        bn = BayesianNetwork(edges)

        cpds = []
        cpd_class = TabularCPD.get_random(variable=class_name, cardinality={class_name: class_state_count})
        cpd_root_node = TabularCPD.get_random(variable=nodes[0], evidence=[class_name], cardinality={nodes[0]: nodes_info[nodes[0]], class_name: class_state_count})
        for i in range(1, len(nodes)):
            cpd_node = TabularCPD.get_random(variable=nodes[i], evidence=[nodes[i-1], class_name], cardinality={nodes[i]: nodes_info[nodes[i]], nodes[i-1]: nodes_info[nodes[i-1]], class_name: class_state_count})
            cpds.append(cpd_node)
        cpds.append(cpd_root_node)
        cpds.append(cpd_class)

        bn.add_cpds(*cpds)

        return cls(bn, nodes, class_name, class_states)
    
    @classmethod
    def initialize_random_dag(cls, nodes_info, class_name, class_states, edge_probability=0.5):
        nodes = list(nodes_info.keys())
        class_state_count = len(class_states)
        edges = cls.generate_random_dag(nodes, class_name, edge_probability=edge_probability)
        nodes_info.update({class_name: class_state_count})
        bn = BayesianNetwork(edges)
        cpds = []
        
        for node in bn.nodes:
            cardinality = {node: nodes_info[node]}
            node_parents = list(bn.get_parents(node))
            for parent in node_parents:
                cardinality.update({parent: nodes_info[parent]})
            if len(node_parents) > 0:
                cpd = TabularCPD.get_random(variable=node, evidence=node_parents, cardinality=cardinality)
            else:
                cpd = TabularCPD.get_random(variable=node, cardinality=cardinality)
            cpds.append(cpd)
        
        del(nodes_info[class_name])
        bn.add_cpds(*cpds)
        
        return cls(bn, nodes, class_name, class_states)
            
    
    @staticmethod
    def generate_random_dag(nodes, class_name, edge_probability):
        edges = []
        all_nodes = nodes[:]
        random.shuffle(all_nodes)
        
        class_childrens = np.random.choice(all_nodes, size=min(5, len(all_nodes)), replace=False)
        for node in class_childrens:
            edges.append((class_name, node))
        
        for node in all_nodes:
            for target in all_nodes:
                if node != target and np.random.random() > edge_probability:
                    try:
                        edges.append((node, target))
                        BayesianNetwork(edges)
                    except:
                        edges.pop()
        return edges
    
    @classmethod
    def initialize_with_bn(cls, bn, nodes_info, class_name, class_states):
        nodes = list(nodes_info.keys())
        return cls(bn, nodes, class_name, class_states)
    
    @classmethod
    def initialize_with_dag(cls, dag, nodes_info, class_name, class_states):
        nodes = list(nodes_info.keys())
        bn = BayesianNetwork(dag.edges())
        return cls(bn, nodes, class_name, class_states)
        
    def set_new_dag(self, dag):
        bn = BayesianNetwork(dag.edges())
        self.bn = bn
    
    def sample(self, current_state: int,  size: int= 1) -> dict:
        inference = VariableElimination(self.bn)
        current_emission_prob = inference.query(variables=self.nodes, evidence={self.class_name: current_state})
        curr_observation = current_emission_prob.sample(1).iloc[0].to_dict()
        return curr_observation
    
    def get_emission_probability(self, evidence: dict) -> np.array:
        inference = VariableElimination(self.bn)
        probs = []
        for state in self.class_states:
            cpd = inference.query(variables=list(evidence.keys()), evidence={self.class_name: state})
            cpd.reduce([(x[0], x[1]) for x in evidence.items()])
            probs.append([cpd.values])
        
        return np.array(probs)
    
    def get_emission_probabilities(self, evidence_df) -> np.ndarray:
        all_probs = []
        for evidence_index in range(evidence_df.shape[0]):
            all_probs.append(self.get_emission_probability(evidence_df.iloc[evidence_index].to_dict()))
        return all_probs
        

class HMMBNetwork(object):
    def __init__(self, pi, T, bn, nodes_info) -> None:
        self.pi = pi
        self.T = T
        self.bn = bn
        self.class_states = bn.class_states
        self.class_name = bn.class_name
        self.nodes_info = nodes_info

    def __repr__(self) -> str:
        pass
    
    @classmethod
    def initialize(cls, class_name: str, class_states_counts: int, nodes_info: dict, edge_probability: float=0.5):
        class_states = list(range(class_states_counts))
        T = ProbabilityMatrix.initialize(class_states, class_states)
        pi = ProbabilityVector.initialize(class_states)
        bn = BNetwork.initialize_random_dag(nodes_info, class_name, class_states, edge_probability=edge_probability)
        return cls(pi, T, bn, nodes_info)
    
    @classmethod
    def initialize_with_dag(cls, dag, class_name: str, class_states_counts: int, nodes_info: dict):
        class_states = list(range(class_states_counts))
        T = ProbabilityMatrix.initialize(class_states, class_states)
        pi = ProbabilityVector.initialize(class_states)
        bn = BNetwork.initialize_with_dag(dag, nodes_info, class_name, class_states)
        return cls(pi, T, bn, nodes_info)
    

    def simulate(self, length):
        s_history = []
        o_history = []

        current_probability = self.pi.values
        current_state = np.random.choice(self.class_states, p = current_probability.flatten())
        current_observation = self.bn.sample(current_state)
        s_history.append(current_state)
        o_history.append(current_observation)

        for i in range(1, length):
            current_probability = self.T.df.loc[current_state]
            current_state = np.random.choice(self.class_states, p=current_probability)
            current_observation = self.bn.sample(current_state)
            s_history.append(current_state)
            o_history.append(current_observation)
        
        return s_history, o_history
    
    def alphas_betas_scaled(self, df):
        emissions = self.bn.get_emission_probabilities(df)

        scaling_factor = np.ones(df.shape[0])
        alphas = np.zeros((df.shape[0], len(self.class_states)))

        alphas[0, :] = self.pi.values * emissions[0].T
        scaling_factor[0] = 1 / (alphas[0,:].sum())
        alphas[0, :] = alphas[0, :] * scaling_factor[0]

        for t in range(1, df.shape[0]):
            alphas[t, :] = (alphas[t-1, :].reshape(1,-1) @ self.T.values) * emissions[t].T
            scaling_factor[t] = 1 / (alphas[t, :].sum())
            alphas[t, :] = alphas[t, :] * scaling_factor[t]
        
        betas = np.zeros((df.shape[0], len(self.class_states)))
        betas[-1, :] = scaling_factor[-1]
        for t in range(df.shape[0]-2, -1, -1):
            betas[t, :] = (self.T.values @ (emissions[t+1] * betas[t+1, :].reshape(-1,1))).reshape(1,-1)
            betas[t, :] = betas[t, :] * scaling_factor[t]
        
        return alphas, betas, scaling_factor
    
    def gammas_digmmas_scaled(self, df: pd.DataFrame, alphas: np.ndarray=None, betas: np.ndarray=None):
        if alphas is None or betas is None:
            alphas, betas, _ = self.alphas_betas_scaled(df)

        emissions = self.bn.get_emission_probabilities(df)
        digammas = np.zeros((df.shape[0] - 1, len(self.class_states), len(self.class_states)))
        gammas = np.zeros((df.shape[0] , len(self.class_states)))
        for t in range(df.shape[0] - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = emissions[t+1].T * betas[t+1, :].reshape(1,-1)
            digammas[t, :, :] = P1 * P2
        gammas[:-1] = digammas.sum(axis=2)
        gammas[-1] = alphas[-1]

        return gammas, digammas
    
    def score(self, df):
        _, _, scaling_factors = self.alphas_betas_scaled(df)
        return -np.log(scaling_factors).sum()
    
    def forward_backward(self, df, alphas=None, betas=None):
        if alphas is None or betas is None:
            alphas, betas, _ = self.alphas_betas_scaled(df)
        max_probability_classes = (alphas * betas).argmax(axis=1)
        return max_probability_classes

    def train(self, df: pd.DataFrame, epoches: int=5, verbose=0) -> list:
        scores = []
        for i in range(1, epoches+1):
            score = self.update(df)
            scores.append(score)
            if verbose>0: print("Epoch {} - Score {}".format(i, score))
            
        return scores

    def train_multiple_observations(self, df_list: list, epoches: int=5, verbose=0) -> list:
        scores = []
        for i in range(1, epoches+1):
            score = self.update_multiple_observations(df_list)
            scores.append(score)
            if verbose>0: print("Epoch {} - Score {}".format(i, score))
            
        return scores
    
    def train_multiple_observations_variable_size(self, df_list: list, epoches: int=5, verbose=0) -> list:
        scores = []
        for i in range(1, epoches+1):
            score = self.update_multiple_observations_variable_size(df_list)
            scores.append(score)
            if verbose>0: print("Epoch {} - Score {}".format(i, score))
            
        return scores
    
    def train_multiple_observations_separate(self, df_list: list, epoches: int=5, verbose : int=0) ->list:
        scores = []
        for i in range(1, epoches+1):
            score = self.update_multiple_observations_separate(df_list)
            scores.append(score)
            if verbose>0: print("Epoch {} - Score {}".format(i, score))
            
        return scores
    
    def update_multiple_observations_separate(self, df_list:list) -> float:
        score_sum = 0
        for df in df_list:
            score_sum += self.update(df)
        
        return score_sum
    
    def fit_bn(self, *args, **kwargs):  
        state_names = {self.class_name: self.class_states}
        for node, node_state_count in self.nodes_info.items():
            state_names[node] = list(range(node_state_count))
        self.bn.bn.fit(*args, **kwargs, state_names=state_names)
        
        
    def set_bn(self, new_dag) -> None:
        bn = BNetwork.initialize_with_dag(new_dag, self.nodes_info, self.class_name, self.class_states)
        self.bn = bn

    def set_pi(self, new_pi) -> None:
        self.pi = new_pi
    
    def set_T(self, new_T) -> None:
        self.T = new_T

    def augment_df(self, df) -> pd.DataFrame:
        alphas, betas, _ = self.alphas_betas_scaled(df)
        class_probabilities = alphas*betas
        class_probabilities = class_probabilities / class_probabilities.sum(axis=1).reshape(-1,1)
        augmented_rows_list = []
        for index, class_probability in enumerate(class_probabilities):
            for state_index, state_value in enumerate(self.class_states):
                d = {self.class_name: state_value, "_weight": class_probability[state_index]}
                d.update(df.iloc[index,:].to_dict())
                augmented_rows_list.append(d)
        
        return pd.DataFrame(augmented_rows_list)
    
    def update(self, df):
        # get alphas, betas, gammas and digammas
        alphas, betas, scaling_factor = self.alphas_betas_scaled(df)
        gammas, digammas = self.gammas_digmmas_scaled(df, alphas, betas)


        # update T and pi
        T = digammas.sum(axis=0) / gammas.sum(axis=0).reshape(-1, 1)   
        T = T / T.sum(axis=1).reshape(-1,1) # rescaling
        T = ProbabilityMatrix.from_numpy(T, self.class_states, self.class_states)

        pi = gammas[0]
        pi = pi / pi.sum()
        pi = ProbabilityVector.from_numpy(pi, self.class_states)

        self.set_pi(pi)
        self.set_T(T)

        augmented_df = self.augment_df(df)

        # EM to estimate bn params
        self.fit_bn(augmented_df, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=10, equivalent_sample_size=1000, weighted=True)

        # construct fictious complete dataset
        augmented_df = self.augment_df(df)

        # find new bn structure using fictious dataset
        est = HillClimbSearch(data = augmented_df.drop(columns=['_weight']))
        new_dag = est.estimate(start_dag=None,
                          max_indegree=5,
                          max_iter=100,
                          show_progress=False,
                          fixed_edges=[(self.class_name, node) for node in list(self.nodes_info.keys())],
                          black_list=[(node, self.class_name) for node in list(self.nodes_info.keys())])
        
        self.set_bn(new_dag)

        # estimate new parameters for new bn
        self.fit_bn(augmented_df, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=10, equivalent_sample_size=1000, weighted=True)
        

        return -np.log(scaling_factor).sum()
    
    
    def update_multiple_observations(self, df_list: list) -> tuple:
        # proper implementation
        alphas_list = []
        gammas_list = []
        digammas_list = []
        scaling_factors_list = []
        
        for obs_index in range(len(df_list)):
            df = df_list[obs_index]
            alphas, betas, scaling_factors = self.alphas_betas_scaled(df)
            gammas, digammas = self.gammas_digmmas_scaled(df, alphas, betas)
            alphas_list.append(alphas)
            gammas_list.append(gammas)
            digammas_list.append(digammas)
            scaling_factors_list.append(scaling_factors)
            
        
        # update pi
        pi = np.array(gammas_list)[:,0,:].mean(axis=0)
        pi = pi / pi.sum()
        pi = ProbabilityVector.from_numpy(pi, self.class_states)

        # update T
        num = 0
        denum = 0
        for j in range(len(df_list)):
            num = num + digammas_list[j].sum(axis=0)
            denum = denum + gammas_list[j].sum(axis=0).reshape(-1,1)

        T = num/denum
        T = T / T.sum(axis=1).reshape(-1,1) # rescaling
        T = ProbabilityMatrix.from_numpy(T, self.class_states, self.class_states)

        self.set_pi(pi)
        self.set_T(T)

        # augmend dataset
        augmented_df_list = []
        for obs_index in range(len(df_list)):
            df = df_list[obs_index]
            augmented_df = self.augment_df(df)
            augmented_df_list.append(augmented_df)

        # update bn params
        self.fit_bn(pd.concat(augmented_df_list), estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=10, equivalent_sample_size=1000, weighted=True)

        # augment dataset again
        augmented_df_list = []
        for obs_index in range(len(df_list)):
            df = df_list[obs_index]
            augmented_df = self.augment_df(df)
            augmented_df_list.append(augmented_df)

        state_names = {self.class_name: self.class_states}
        for node, node_state_count in self.nodes_info.items():
            state_names[node] = list(range(node_state_count))
        
        train_struc_df = pd.concat(augmented_df_list).drop(columns=['_weight'])

        from pgmpy.estimators import K2Score, BDeuScore, BicScore, BDsScore
        scoring_method = BDeuScore(data=train_struc_df, state_names=state_names, equivalent_sample_size=2000)
        # find new tan structure using fictious dataset
        est = HillClimbSearch(data = train_struc_df)
        new_dag = est.estimate(start_dag=None,
                          scoring_method=scoring_method,
                          tabu_length=5,
                          show_progress=False,
                          fixed_edges=[(self.class_name, node) for node in list(self.nodes_info.keys())],
                          black_list=[(node, self.class_name) for node in list(self.nodes_info.keys())])
        
        self.set_bn(new_dag)

        # find new parameters for new tan
        self.fit_bn(pd.concat(augmented_df_list), estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=10, equivalent_sample_size=1000, weighted=True)

        return -np.log(scaling_factors_list).sum()
    
    
    def update_multiple_observations_variable_size(self, df_list: list) -> tuple:
        # proper implementation
        alphas_list = []
        gammas_list = []
        digammas_list = []
        scaling_factors_list = []
        
        for obs_index in range(len(df_list)):
            df = df_list[obs_index]
            alphas, betas, scaling_factors = self.alphas_betas_scaled(df)
            gammas, digammas = self.gammas_digmmas_scaled(df, alphas, betas)
            alphas_list.append(alphas)
            gammas_list.append(gammas)
            digammas_list.append(digammas)
            scaling_factors_list.append(scaling_factors)
            
        
        # update pi
        pi = np.array(gammas_list[0][0,:])
        for gamma in gammas_list[1:]:
            pi+=np.array(gamma)[0,:]
        pi = pi / len(gammas_list)
        pi = pi / pi.sum()
        pi = ProbabilityVector.from_numpy(pi, self.class_states)

        # update T
        num = 0
        denum = 0
        for j in range(len(df_list)):
            num = num + digammas_list[j].sum(axis=0)
            denum = denum + gammas_list[j].sum(axis=0).reshape(-1,1)

        T = num/denum
        T = T / T.sum(axis=1).reshape(-1,1) # rescaling
        T = ProbabilityMatrix.from_numpy(T, self.class_states, self.class_states)

        self.set_pi(pi)
        self.set_T(T)

        # augmend dataset
        augmented_df_list = []
        for obs_index in range(len(df_list)):
            df = df_list[obs_index]
            augmented_df = self.augment_df(df)
            augmented_df_list.append(augmented_df)

        # update bn params
        self.fit_bn(pd.concat(augmented_df_list), estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=10, equivalent_sample_size=1000, weighted=True)

        # augment dataset again
        augmented_df_list = []
        for obs_index in range(len(df_list)):
            df = df_list[obs_index]
            augmented_df = self.augment_df(df)
            augmented_df_list.append(augmented_df)

        # find new bn structure using fictious dataset
        est = HillClimbSearch(data = pd.concat(augmented_df_list).drop(columns=['_weight']))
        new_dag = est.estimate(start_dag=None,
                          max_indegree=5,
                          max_iter=100,
                          show_progress=False,
                          fixed_edges=[(self.class_name, node) for node in list(self.nodes_info.keys())],
                          black_list=[(node, self.class_name) for node in list(self.nodes_info.keys())])
        
        
        self.set_bn(new_dag)

        # find new parameters for new bn
        self.fit_bn(pd.concat(augmented_df_list), estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=10, equivalent_sample_size=1000, weighted=True)

        log_score = 0
        for scaling_factors in scaling_factors_list:
            log_score+= -np.log(scaling_factors).sum()
        
        return log_score


### working version

# from pgmpy.estimators import K2Score, BDeuScore, BicScore, BDsScore
#         scoring_method = BDeuScore(data=train_struc_df, state_names=state_names, equivalent_sample_size=2000)
#         # find new tan structure using fictious dataset
#         est = HillClimbSearch(data = train_struc_df)
#         new_dag = est.estimate(start_dag=None,
#                           scoring_method=scoring_method,
#                           tabu_length=5,
#                           show_progress=False,
#                           fixed_edges=[(self.class_name, node) for node in list(self.nodes_info.keys())],
#                           black_list=[(node, self.class_name) for node in list(self.nodes_info.keys())])