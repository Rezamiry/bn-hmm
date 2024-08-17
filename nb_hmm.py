import numpy as np
import pandas as pd
from itertools import product
from functools import reduce


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


class HiddenMarkovModel:
    """
    Base class for working with hidden markov model.
    This class provides the following functionalities:
        - simulate: simulating observations and state transitions given model parameters
        - score, foward: liklihood of a sequence of observations given the model
        - forward_backward: most likly states given observations and model
        - viterbi: most likly sequence of states given observations and model
        - train: train the model given observations (and known states) with baum-welch

    Parameters:
        T: transition matrix A
        E: emission matrix B
        pi: initial state probability
        observables: list of obsevables of emissions
    """
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
        self._score_init = - np.Inf
        self.score_history = []

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            # TODO change this mechanism to something better for state transition tuoples
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(
                map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(
                map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score

    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)
                         @ self.T.values) * self.E[observations[t]].T # apply a function instead of using self.E, also the function need to take the state and observation as input and return in matrix format
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]]
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas

    def forward(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())

    def forward_backward(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))
    
    def viterbi(self, observations: list) -> list:
        K = len(self.states)
        # Initialize the priors with default (uniform dist) if not given by caller
        T = len(observations)
        T1 = np.empty((K, T), 'd')
        T2 = np.empty((K, T), 'B')

        # Initilaize the tracking tables from first observation
        T1[:, 0] = self.pi.values * self.E[observations[0]].reshape(1,-1)
        T1[:, 0] = T1[:, 0] / T1[:, 0].sum()
        T2[:, 0] = 0

        # Iterate throught the observations updating the tracking tables
        for i in range(1, T):
            T1[:, i] = np.max(T1[:, i - 1] *  self.T.values.T * self.E[observations[i]].T, 1)
            T1[:, i] = T1[:, i] / T1[:, i].sum()
            T2[:, i] = np.argmax(T1[:, i - 1] *  self.T.values.T, 1)
            
        # Build the output, optimal model trajectory
        x = np.empty(T, 'B')
        x[-1] = np.argmax(T1[:, T - 1])
        for i in reversed(range(1, T)):
            x[i - 1] = T2[x[i], i]
            
        pred = [self.states[i] for i in x]
        return pred
        

    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.forward(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas
    
    def _alphas_betas_scaled(self, observations: list, observed_states: list=None) -> (np.ndarray, np.ndarray, np.ndarray):
        scaling_factor = np.ones(len(observations))
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        # scaling
        if observed_states is not None and observed_states[0]:
            alphas[0, :] = 1/1e50
            alphas[0, self.states.index(observed_states[0])] = 1
        scaling_factor[0] = 1 / (alphas[0,:].sum())
        alphas[0, :] = alphas[0, :] * scaling_factor[0]
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) 
                         @ self.T.values) * self.E[observations[t]].T
            if observed_states is not None and observed_states[t]:
                alphas[t, :] = 1/1e50
                alphas[t, self.states.index(observed_states[t])] = 1
            scaling_factor[t] = 1 / (alphas[t, :].sum())
            alphas[t, :] = alphas[t, :] * scaling_factor[t]
        
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = scaling_factor[-1]
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
            if observed_states is not None and observed_states[t]:
                betas[t, :] = 1/1e50
                betas[t, self.states.index(observed_states[t])] = 1
            betas[t, :] = betas[t, :] * scaling_factor[t]
        return (alphas, betas, scaling_factor)
    
    
    def _gamma_digammas_scaled_ab(self, observations: list, observed_states:list=None) -> (np.ndarray, np.ndarray):
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))
        gammas = np.zeros((L , N))

        alphas, betas, _ = self._alphas_betas_scaled(observations, observed_states)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2
        
        
        gammas[:-1] = digammas.sum(axis=2)
        gammas[-1] = alphas[-1]
        
        return (gammas, digammas)
    
    
    
    def simulate(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length)
        o_history = [0] * (length)

        curr_prob = self.pi.values
        curr_state = np.random.choice(self.states, p=curr_prob.flatten())
        curr_obs = np.random.choice(self.observables, p=self.E.df.loc[curr_state])
        s_history[0] = curr_state
        o_history[0] = curr_obs

        for t in range(1, length):
            curr_prob = self.T.df.loc[curr_state]
            curr_state = np.random.choice(self.states, p=curr_prob)
            curr_obs = np.random.choice(self.observables, p=self.E.df.loc[curr_state])
            s_history[t] = curr_state
            o_history[t] = curr_obs
        
        return o_history, s_history
    
    def train(self, observations: list, epochs: int, observed_states: list=None, tol=None):
        self._score_init = - np.Inf
        self.score_history = (epochs) * [-np.Inf]
        early_stopping = isinstance(tol, (int, float))
        
        for epoch in range(1, epochs + 1):
            score = self.update(observations, observed_states)
            print_interval = int(epochs/10)
            # if epoch % print_interval== 0:
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / abs(score) < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch-1] = score
            
    def update(self, observations: list, observed_states: list=None) -> float:
        
        alphas, betas, scaling_factors = self._alphas_betas_scaled(observations, observed_states)
        gammas, digammas = self._gamma_digammas_scaled_ab(observations, observed_states)    

        L = len(alphas)
        obs_idx = [self.observables.index(x) \
                  for x in observations]
        
        pi = gammas[0]
        
        capture = np.zeros((L, len(self.states), len(self.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0

        
        T = digammas.sum(axis=0) / gammas[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gammas[:, :, np.newaxis]).sum(axis=0) / gammas.sum(axis=0).reshape(-1, 1)
        
        
        T = T / T.sum(axis=1).reshape(-1,1)
        E = E / E.sum(axis=1).reshape(-1,1)
        pi = pi / pi.sum()
        
        self.pi = ProbabilityVector.from_numpy(pi, self.states)
        self.T = ProbabilityMatrix.from_numpy(T, self.states, self.states)
        self.E = ProbabilityMatrix.from_numpy(E, self.states, self.observables)
        
        score = - np.log(scaling_factors).sum()
        return score
    


class NaiveBayesHMM(object):
    """
    This class extend HMM class with support for naive bayes observations and 
    multiple instances
    This class provides the following functionalities:
        - simulate: simulating observations and state transitions given model parameters
        - score, foward: liklihood of a sequence of observations given the model
        - forward_backward: most likly states given observations and model
        - viterbi: most likly sequence of states given observations and model
        - train: train the model given observations (and known states) with baum-welch
        - train_multiple_observations: train using multiple sequence of naive bayes observations
    
    Parameters:
        T: transition matrix A
        E: list of emission matrixes Bs
        pi: initial state probability
        observables: list of obsevables of each emission variables
    """
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # list of emission matrixes B
        self.pi = pi
        self.states = pi.states
        self.observables = [e.observables for e in self.E]
        self._score_init = - np.Inf
        self.score_history = []
    
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = [ProbabilityMatrix.initialize(states, observable) for observable in observables]
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)
    
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))
    

    def _alphas_betas_scaled(self, observations: list, observed_states: list=None) -> (np.ndarray, np.ndarray, np.ndarray):
        scaling_factor = np.ones(len(observations[0]))
        alphas = np.zeros((len(observations[0]), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[0][observations[0][0]].T
        # iterate over all 0th observation of all observations
        for i in range(1, len(observations)): 
            alphas[0, :] = alphas[0, :] * self.E[i][observations[i][0]].T
        
        # scaling
        if observed_states is not None and not pd.isnull(observed_states[0]):
            alphas[0, :] = 1/1e50
            alphas[0, self.states.index(observed_states[0])] = 1
        
        scaling_factor[0] = 1 / (alphas[0,:].sum())
        alphas[0, :] = alphas[0, :] * scaling_factor[0]

        for t in range(1, len(observations[0])):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) 
                         @ self.T.values) * self.E[0][observations[0][t]].T
            # iterate over all ith observation of all observations
            for i in range(1, len(observations)): 
                alphas[t, :] = alphas[t, :] * self.E[i][observations[i][t]].T
            if observed_states is not None and not pd.isnull(observed_states[t]):
                alphas[t, :] = 1/1e50
                alphas[t, self.states.index(observed_states[t])] = 1
            scaling_factor[t] = 1 / (alphas[t, :].sum())
            alphas[t, :] = alphas[t, :] * scaling_factor[t]
        
        betas = np.zeros((len(observations[0]), len(self.states)))
        betas[-1, :] = scaling_factor[-1]
        for t in range(len(observations[0]) - 2, -1, -1):
            # iterate over all ith observation of all observations
            temp = self.E[0][observations[0][t+1]]
            for i in range(1, len(observations)):
                temp = temp * self.E[i][observations[i][t+1]]

            betas[t, :] = (self.T.values @ (temp \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
            
            if observed_states is not None and not pd.isnull(observed_states[t]):
                betas[t, :] = 1/1e50
                betas[t, self.states.index(observed_states[t])] = 1
            betas[t, :] = betas[t, :] * scaling_factor[t]
        return (alphas, betas, scaling_factor)
    
    def score(self, observations: list, observed_states: list=None):
        (_, _, scaling_factor) = self._alphas_betas_scaled(observations, observed_states)
        score = - np.log(scaling_factor).sum()
        return score


    def _gamma_digammas_scaled_ab(self, observations: list, observed_states:list=None) -> (np.ndarray, np.ndarray):
        L, N = len(observations[0]), len(self.states)
        digammas = np.zeros((L - 1, N, N))
        gammas = np.zeros((L , N))

        alphas, betas, _ = self._alphas_betas_scaled(observations, observed_states)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[0][observations[0][t + 1]].T * betas[t + 1].reshape(1, -1)
            for i in range(1, len(observations)):
                P2 = self.E[i][observations[i][t + 1]].T * P2
            digammas[t, :, :] = P1 * P2
        
        
        gammas[:-1] = digammas.sum(axis=2)
        gammas[-1] = alphas[-1]
        
        return (gammas, digammas)
    
    def simulate(self, length: int) -> (list, list):
        s_history = [0] * (length)
        o_history = [[]  for i in range(len(self.E))]

        curr_prob = self.pi.values
        curr_state = np.random.choice(self.states, p=curr_prob.flatten())
        for i in range(len(self.E)):
            o_history[i].append(np.random.choice(self.observables[i], p=self.E[i].df.loc[curr_state]))

        s_history[0] = curr_state

        for t in range(1, length):
            curr_prob = self.T.df.loc[curr_state]
            curr_state = np.random.choice(self.states, p=curr_prob)
            for i in range(len(self.E)):
                o_history[i].append(np.random.choice(self.observables[i], p=self.E[i].df.loc[curr_state]))
            s_history[t] = curr_state
            
        return o_history, s_history
    


    def train(self, observations: list, epochs: int, observed_states: list=None, tol=None, verbose: int=0):
        self._score_init = - np.Inf
        self.score_history = (epochs) * [-np.Inf]
        early_stopping = isinstance(tol, (int, float))
        
        for epoch in range(1, epochs + 1):
            score = self.update(observations, observed_states)
            print_interval = int(epochs/10)
            # if epoch % print_interval== 0:
            if verbose>0 : print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / abs(score) < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch-1] = score
    
    def update(self, observations: list, observed_states: list=None) -> float:
        
        alphas, betas, scaling_factors = self._alphas_betas_scaled(observations, observed_states)
        gammas, digammas = self._gamma_digammas_scaled_ab(observations, observed_states)    

        pi = gammas[0]

        E_list = []
        L = len(alphas)
        for i in range(len(self.E)):
            obs_idx = [self.observables[i].index(x) for x in observations[i]]
            capture = np.zeros((L, len(self.states), len(self.observables[i])))
            for t in range(L):
                capture[t, :, obs_idx[t]] = 1.0
            E = (capture * gammas[:, :, np.newaxis]).sum(axis=0) / gammas.sum(axis=0).reshape(-1, 1)
            E_list.append(E)
        
        T = digammas.sum(axis=0) / gammas.sum(axis=0).reshape(-1, 1)
        
        # rescaling
        T = T / T.sum(axis=1).reshape(-1,1)
        for i in range(len(E_list)):
            E_list[i] = E_list[i] / E_list[i].sum(axis=1).reshape(-1,1)
    
        pi = pi / pi.sum()
        
        self.pi = ProbabilityVector.from_numpy(pi, self.states)
        self.T = ProbabilityMatrix.from_numpy(T, self.states, self.states)
        self.E = [ProbabilityMatrix.from_numpy(E_list[i], self.states, self.observables[i]) for  i \
                  in range(len(E_list))]
        
        score = - np.log(scaling_factors).sum()
        return score
    
    def forward_backward(self, observations: list) -> list:
        alphas, betas, _ = self._alphas_betas_scaled(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))

    
    def train_multiple_observations(self, observations_list: list, epochs: int, observed_states_list: list=None, tol=None, verbose: int=0):
        self._score_init = - np.Inf
        self.score_history = (epochs) * [-np.Inf]
        early_stopping = isinstance(tol, (int, float))
        
        for epoch in range(1, epochs + 1):
            score = self.update_multiple_observations(observations_list, observed_states_list)
            print_interval = int(epochs/10)
            # if epoch % print_interval== 0:
            if verbose>0: print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / abs(score) < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch-1] = score

    def update_multiple_observations(self, observations_list: list, observed_states_list: list=None) -> float:
        
        alphas_list = []
        gammas_list = []
        digammas_list = []
        scaling_factors_list = []
        for obs_index in range(len(observations_list)):
            observations = observations_list[obs_index]
            observed_states = None
            if observed_states_list is not None:
                observed_states = observed_states_list[obs_index]
            
            alphas, betas, scaling_factors = self._alphas_betas_scaled(observations, observed_states)
            gammas, digammas = self._gamma_digammas_scaled_ab(observations, observed_states)    
            alphas_list.append(alphas)
            gammas_list.append(gammas)
            digammas_list.append(digammas)
            scaling_factors_list.append(scaling_factors)

        # update pi
        # pi = np.array(gammas_list)[:,0,:].mean(axis=0)
        pi = np.array(gammas_list[0][0,:])
        for gamma in gammas_list[1:]:
            pi+=np.array(gamma)[0,:]
        pi = pi / len(gammas_list)
        pi = pi / pi.sum()

        # update E
        E_list = []
        
        for i in range(len(self.E)):
            num = 0
            denum = 0
            # go through each observation and sum num and denum
            for j in range(len(observations_list)):
                L = len(alphas_list[j])
                observations = observations_list[j]
                gammas = gammas_list[j]
                obs_idx = [self.observables[i].index(x) for x in observations[i]]
                capture = np.zeros((L, len(self.states), len(self.observables[i])))
                for t in range(L):
                    capture[t, :, obs_idx[t]] = 1.0
                num = num + (capture * gammas[:, :, np.newaxis]).sum(axis=0)
                denum = denum + gammas.sum(axis=0).reshape(-1, 1)

            E = num / denum
            E_list.append(E)


        # update T
        num = 0
        denum = 0
        for j in range(len(observations_list)):
            num = num + digammas_list[j].sum(axis=0)
            denum = denum + gammas_list[j].sum(axis=0).reshape(-1,1)

        T = num/denum
        T = T / T.sum(axis=1).reshape(-1,1)
        for i in range(len(E_list)):
            E_list[i] = E_list[i] / E_list[i].sum(axis=1).reshape(-1,1)
    
        
        self.pi = ProbabilityVector.from_numpy(pi, self.states)
        self.T = ProbabilityMatrix.from_numpy(T, self.states, self.states)
        self.E = [ProbabilityMatrix.from_numpy(E_list[i], self.states, self.observables[i]) for  i \
                  in range(len(E_list))]
        
        log_score = 0
        for scaling_factors in scaling_factors_list:
            log_score+= -np.log(scaling_factors).sum()
            
        return log_score