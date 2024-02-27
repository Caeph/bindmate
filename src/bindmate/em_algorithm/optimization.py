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


# def numerical_argmax_(observed_vector, model_qs, probafunc, param_bounds, weight_bounds=(0.1, 100)):
#     def objective(params, pseudocount=10e-10):
#         weight = params[0]
#         fun_params = params[1:]
#         argmaxing = model_qs * np.log(probafunc(observed_vector, fun_params) + pseudocount)
#         return -np.sum(argmaxing)
#
#     bounds = (weight_bounds, *param_bounds)
#
#     minimizing = optimize.dual_annealing(objective, bounds)
#     return minimizing.x

class ProbabilityModelEnsemble:
    def __init__(self, models):
        self.models = models

    def calculate_probability(self, z, observed_values):
        return self.models[z].calculate_probability(observed_values)

    def argmax_params(self, qs, observed_values, new_priors):
        new_theta = {}
        for z in self.models.keys():
            new_theta[z] = self.models[z].argmax_for_parameters(qs[z], observed_values)
        return new_theta

    def __len__(self):
        return len(self.models)

    def get_models(self):
        return self.models

    def set_params(self, parameters, z):
        self.models[z].set_parameters(parameters)


class WeightedModelEnsemble(ProbabilityModelEnsemble):
    # KL divergence as the weight - no new parameters

    def __init__(self, models):
        super().__init__(models)
        self.weights = None
        # initialize weights TODO

    def calculate_probability(self, z, observed_values, pseudocount=1e-10):
        # init
        if self.weights is None:
            self.weights = {z: {m: 1 for m in range(observed_values.shape[1])} for z in self.models.keys()}

        model = self.models[z]

        result = np.zeros(len(observed_values))
        for i in range(observed_values.shape[1]):
            p_func = model.optimization_info[i]['proba']
            param = model.parameters[i]
            x = pd.Series(observed_values[:, i])
            p = np.log(
                x.swifter.progress_bar(False).apply(lambda xi: p_func(xi, param) + pseudocount)
            )

            # weighting
            p = p * self.weights[z][i]

            result = result + p
            if type(result) is pd.Series:
                result = result.values

        return np.exp(result)

    def argmax_params(self, qs, observed_values, new_priors):
        # one call of numerical solving U
        def KL_divergence_weight(z, m, p_xmi, pseudocount=1e-9):
            p_xi = np.sum(p_xmi[:, :, m], axis=0) + pseudocount
            p_xi_given_mz = p_xmi[z, :, m]

            w_avg = np.sum((p_xi_given_mz * new_priors[z]) * np.log(
                (p_xi_given_mz / p_xi) + pseudocount
            ))
            bottom_entropy = np.sum(p_xi * np.log(p_xi))+pseudocount

            return w_avg / bottom_entropy

        def all_probas(params, pseudocount=1e-10):
            # xmi calculation
            p_xmi = np.stack([np.zeros_like(observed_values) for z in self.models.keys()]).astype(float)
            for m in range(observed_values.shape[1]):
                x = pd.Series(observed_values[:, m])
                for z in self.models.keys():
                    model = self.models[z]
                    param = params[z][m]
                    p_func = model.optimization_info[m]['proba']
                    p_xmi[z, :, m] += x.swifter.progress_bar(False).apply(lambda xi: p_func(xi, param) + pseudocount)
            return p_xmi

        def calculate_weights(p_xmi, minimal_weight=None):
            weights = {}
            for z in self.models.keys():
                weights[z] = {}
                for m in self.weights[z].keys():
                    weight_mz = KL_divergence_weight(z, m, p_xmi)
                    weights[z][m] = weight_mz

            # normalize weights
            Z = np.sum([weights[z][m] for z in self.models.keys() for m in self.weights[z].keys()])
            no_z = len(self.models.keys())
            no_m = observed_values.shape[1]
            Z = (Z / (no_z * no_m))
            for z in self.models.keys():
                for m in self.weights[z].keys():
                    weights[z][m] = weights[z][m] / Z
                    if minimal_weight is not None:
                        weights[z][m] = np.fmax(weights[z][m], minimal_weight)
            return weights

        def full_objective(params):
            # reorder params to dictionary if needed
            params = flat_array_to_params_dict(params)

            p_xmi = all_probas(params)
            weights = calculate_weights(p_xmi)

            # do calculation
            result = 0
            for z in self.models.keys():
                for m in self.weights[z].keys():
                    p = p_xmi[z, :, m]
                    weight_mz = weights[z][m]
                    current_res = weight_mz * np.sum(p * qs[z])
                    result = result + current_res
            return - result

        init_params = []
        about_params = []
        param_bounds = []
        for z in range(len(self.models)):
            for m in range(observed_values.shape[1]):
                init_params.extend(self.models[z].parameters[m])
                param_bounds.extend([check_bounds(bounds, observed_values[:, m]) for bounds in self.models[z].optimization_info[m]['params_bounds']])

                about_params.append(len(self.models[z].parameters[m]))

        def flat_array_to_params_dict(params_vector):
            index = 0
            params = {}
            about_i = 0
            for z in range(len(self.models)):
                params[z] = []
                for m in range(observed_values.shape[1]):
                    l = about_params[about_i]
                    params[z].append(params_vector[index:index+l])
                    about_i += 1
                    index += l
            return params

        # numerically minimize objective function
        minimizing = optimize.minimize(full_objective, np.array(init_params), bounds=param_bounds)
        # minimizing = optimize.dual_annealing(full_objective, bounds=param_bounds)
        best_params = flat_array_to_params_dict(minimizing.x)

        best_p_xmi = all_probas(best_params)
        self.weights = calculate_weights(best_p_xmi, minimal_weight=1e-6)
        print(f"Achieved best weights: {self.weights}")

        return best_params




def numerical_argmax_func(observed_vector, model_qs, probafunc, param_bounds):
    def objective(params):
        a = model_qs * np.log(probafunc(observed_vector, params) + 1e-6)

        return -np.sum(a[~np.isnan(a)])

    minimizing = optimize.dual_annealing(objective, param_bounds)
    return minimizing


def check_bounds(b, observed):
    if b is not None:
        return b

    return 0 + 10e-6, max(observed)


class WeightedProbabilityModel(ProbabilityModel):

    def __init__(self, z, metrics, get_params_from_matched=False):
        super().__init__(z, metrics, get_params_from_matched)
        # self.parameters = [[1, *x] for x in self.parameters]

    def calculate_probability(self, observed_values, pseudocount=10e-10):
        # Pr[observed | params, z] for every observation
        # this model should be rewritten based on the particular distribution

        result = np.zeros(len(observed_values))
        for i in range(observed_values.shape[1]):
            p_func = self.optimization_info[i]['proba']
            param = self.parameters[i]  # [1:]  # without weight
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
        print(f"running argmax for params in weighted proba model {self.identificator}")

        new_parameters = []
        for i, optimizer in enumerate(self.optimization_info):
            p_func = optimizer['proba']
            bounds = optimizer['params_bounds']
            bounds = [check_bounds(b, observed_values[:, i]) for b in bounds]
            solution = numerical_argmax_func(observed_values[:, i], model_qs, p_func, bounds)
            # parameters of the function
            new_parameters.append(solution.x)

        return new_parameters


#  assumes metrics are conditionally independent
class EMOptimizer:
    # https://courses.csail.mit.edu/6.867/wiki/images/b/b5/Em_tutorial.pdf
    def __init__(self, possible_latent, priors, models, weighted=False):
        self.z = possible_latent
        # self.models = models  # dictionary z: model[z]
        if weighted:
            self.models = WeightedModelEnsemble(models)
        else:
            self.models = ProbabilityModelEnsemble(models)
        self.priors = priors  # dictionary z: prior[z]
        self.weighted = weighted

    def __e_step(self, observed_values):
        print("E-step started")
        qs = {}
        # for every z, calculate PRIOR[z] * Pr[x | z, params]
        for z in self.z:
            #  q = self.models[z].calculate_probability(observed_values) * self.priors[z]
            q = self.models.calculate_probability(z, observed_values) * self.priors[z]
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

        # new_theta = {}
        # for z in self.z:
        #     new_theta[z] = self.models[z].argmax_for_parameters(qs[z], observed_values)
        new_theta = self.models.argmax_params(qs, observed_values, new_priors)

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
        old_parameters = [self.models.get_models()[z].parameters for z in self.z]
        old_parameters.extend([self.priors[z] for z in self.z])
        self.__record(old_parameters, colector=parameter_colector)

        for i in range(max_step):
            qs = self.__e_step(observed_values)
            new_priors, new_theta = self.__m_step(qs, observed_values)

            self.priors = new_priors
            for z in self.z:
                # self.models[z].set_parameters(new_theta[z])
                self.models.set_params(new_theta[z], z)

            new_parameters = [self.models.get_models()[z].parameters for z in self.z]
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

        return self.models.get_models()  # recorded_parameters
