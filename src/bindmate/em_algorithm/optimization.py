import numpy as np
import pandas as pd
from scipy import optimize


def random_shift(initial_params):
    shifted_params = []
    for item in initial_params:
        new_param = item + np.random.normal(loc=0,
                                            scale=np.fmin(
                                                0.5, item / 100
                                            ))
        shifted_params.append(new_param)
    return shifted_params


class ProbabilityModel:
    def __init__(self, z, metrics, get_params_from_matched=False):
        self.identificator = z
        if get_params_from_matched:
            optim_info = [x.get_optimalization_info(1) for x in metrics]
            self.parameters = [random_shift(params['initial_parameters']) for params in optim_info]
        else:
            optim_info = [x.get_optimalization_info(z) for x in metrics]
            self.parameters = [params['initial_parameters'] for params in optim_info]
        self.optimization_info = optim_info  # the same order as metrics

    def set_parameters(self, new_parameters):
        self.parameters = new_parameters

    def calculate_probability(self, observed_values, pseudocount=10e-10):
        # Pr[observed | params, z] for every observation
        # this model should be rewritten based on the particular distribution

        result = np.zeros(len(observed_values))
        for i in range(observed_values.shape[1]):
            p_func = self.optimization_info[i]['proba']
            param = self.parameters[i]
            x = pd.Series(observed_values[:, i])
            p = np.log(
                x.swifter.apply(lambda xi: p_func(xi, param) + pseudocount)
            )  # considerably faster -- uses dask

            # p = np.log(
            #     np.vectorize(lambda xi: p_func(xi, param) + pseudocount)(x)
            # )
            # TODO nan check

            result = result + p
            if type(result) is pd.Series:
                result = result.values

        return np.exp(result)

    def argmax_for_parameters(self, model_qs, observed_values):
        # calculate parameters for the next iteration
        new_parameters = []
        for i, optimizer in enumerate(self.optimization_info):
            new_theta = optimizer['argmax'](model_qs, observed_values[:, i])
            new_parameters.append(new_theta)

        return new_parameters


def argmax_for_weighted_proba(observed_vector, model_qs, probafunc, param_bounds, weight_bounds=(0.1, 100)):
    def objective(params, pseudocount=10e-10):
        weight = params[0]
        fun_params = params[1:]
        argmaxing = model_qs * weight * np.log(probafunc(observed_vector, fun_params) + pseudocount)
        return -np.sum(argmaxing)

    bounds = (weight_bounds, *param_bounds)

    minimizing = optimize.dual_annealing(objective, bounds)
    return minimizing.x


class WeightedProbabilityModel(ProbabilityModel):

    def __init__(self, z, metrics, get_params_from_matched=False):
        super().__init__(z, metrics, get_params_from_matched)
        self.parameters = [[1, *x] for x in self.parameters]


    def calculate_probability(self, observed_values, pseudocount=10e-10):
        # Pr[observed | params, z] for every observation
        # this model should be rewritten based on the particular distribution

        result = np.zeros(len(observed_values))
        for i in range(observed_values.shape[1]):
            p_func = self.optimization_info[i]['proba']
            param = self.parameters[i][1:]  # without weight
            x = pd.Series(observed_values[:, i])
            p = np.log(
                x.swifter.apply(lambda xi: p_func(xi, param) + pseudocount)
            )  # considerably faster -- uses dask
            result = result + p
            if type(result) is pd.Series:
                result = result.values

        return np.exp(result)


    # does not require to do the theoretical gradient calculating, proceeds numerically
    # the bounds could be a weakness - TODO maybe redefine?
    def argmax_for_parameters(self, model_qs, observed_values):
        print("running argmax for params in weighted proba model")
        new_parameters = []
        for i, optimizer in enumerate(self.optimization_info):
            p_func = optimizer['proba']
            bounds = optimizer['params_bounds']
            new_theta = argmax_for_weighted_proba(observed_values[:, i], model_qs, p_func, bounds)
            # params bounds definition
            new_parameters.append(new_theta)

        return new_parameters


#  assumes metrics are independent
class EMOptimizer:
    # https://courses.csail.mit.edu/6.867/wiki/images/b/b5/Em_tutorial.pdf
    def __init__(self, possible_latent, priors, models):
        self.z = possible_latent
        self.models = models  # dictionary z: model[z]
        self.priors = priors  # dictionary z: prior[z]

    def __e_step(self, observed_values):
        print("E-step started")
        qs = {}
        # for every z, calculate PRIOR[z] * Pr[x | z, params]
        for z in self.z:
            q = self.models[z].calculate_probability(observed_values) * self.priors[z]
            qs[z] = q

        # bottom = sum PRIOR[z] * Pr[x | z, params] over all z
        bottom = np.zeros(len(observed_values))
        for z in self.z:
            bottom = bottom + qs[z]

        # calculate PRIOR[z] * Pr[x | z, params] / (sum PRIOR[z] * Pr[x | z, params] over all z) for every z
        for z in self.z:
            qs[z] = qs[z] / bottom

        print("E-step done")
        return qs

    def __m_step(self, qs, observed_values, thr=0.001):
        print("M-step started")
        new_priors = {}
        for z in self.z:
            p = np.mean(qs[z])
            # p = np.fmax(thr, p)
            # p = np.fmin(1 - thr, p)
            new_priors[z] = p

        new_theta = {}
        for z in self.z:
            new_theta[z] = self.models[z].argmax_for_parameters(qs[z], observed_values)

        print("M-step done")
        return new_priors, new_theta

    def __record(self, parameters, colector, sep=';', subsep=';'):
        if colector is None:
            return

        def flatten_to_string(l):
            flat_list = [str(item) for sublist in l for item in sublist]
            return np.array(flat_list)

        # unmatched, matched, prior0, prior1 = parameters
        # unmatched = subsep.join(flatten_to_string(unmatched))
        # matched = subsep.join(flatten_to_string(matched))
        # print(sep.join([unmatched, matched, str(prior0), str(prior1)]), file=colector)
        priors = [str(x) for x in parameters[-len(self.models):]]
        model_params = [subsep.join(flatten_to_string(x)) for x in parameters[:-len(self.models)]]
        print(sep.join([*model_params, *priors]), file=colector)

    def check_convergence(self, old_parameters, new_parameters, tol):
        def flatten(l):
            flat_list = [item for sublist in l for item in sublist]
            return np.array(flat_list)

        new_params, new_priors = new_parameters[:-len(self.models)], new_parameters[-len(self.models):]
        old_params, old_priors = old_parameters[:-len(self.models)], old_parameters[-len(self.models):]

        # compare priors:
        for o, n in zip(old_priors, new_priors):
            if np.abs(o - n) > tol:
                return False

        # compare models
        for o, n in zip(old_params, new_params):
            if not np.allclose(flatten(o), flatten(n), tol):
                return False

        # new_unmatched, new_matched, new_prior0, new_prior1 = new_parameters
        # old_unmatched, old_matched, old_prior0, old_prior1 = old_parameters
        #
        # # compare priors
        # for o, n in zip([old_prior0, old_prior1], [new_prior0, new_prior1]):
        #     if np.abs(o - n) > tol:
        #         return False
        #
        # new_unmatched = flatten(new_unmatched)
        # new_matched = flatten(new_matched)
        # old_unmatched = flatten(old_unmatched)
        # old_matched = flatten(old_matched)
        #
        # for o, n in zip([old_matched, old_unmatched], [new_matched, new_unmatched]):
        #     if not np.allclose(o, n, tol):
        #         return False
        #
        # return True

    def optimize(self, observed_values, max_step, tolerance, parameter_colector=None):
        old_parameters = [self.models[z].parameters for z in self.z]
        old_parameters.extend([self.priors[z] for z in self.z])
        self.__record(old_parameters, colector=parameter_colector)

        for i in range(max_step):
            qs = self.__e_step(observed_values)
            new_priors, new_theta = self.__m_step(qs, observed_values)

            self.priors = new_priors
            for z in self.z:
                self.models[z].set_parameters(new_theta[z])

            new_parameters = [self.models[z].parameters for z in self.z]
            new_parameters.extend([self.priors[z] for z in self.z])

            print(f"ITERATION {i}")
            print(f"OLD: {old_parameters}")
            print(f"NEW: {new_parameters}")

            convergence = self.check_convergence(old_parameters, new_parameters, tolerance)
            if convergence:
                print("CONVERGENCE")
                break

            # recorded_parameters.append(old_parameters)
            self.__record(new_parameters, colector=parameter_colector)
            old_parameters = new_parameters

        return self.models  # recorded_parameters
