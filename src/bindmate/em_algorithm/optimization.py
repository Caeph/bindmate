import numpy as np
import pandas as pd
import swifter


class ProbabilityModel: 
    def __init__(self, z, metrics):
        self.identificator = z
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
                x.swifter.progress_bar(False).apply(lambda xi: p_func(xi, param) + pseudocount)
            )
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
    

class EMOptimizer:
    # https://courses.csail.mit.edu/6.867/wiki/images/b/b5/Em_tutorial.pdf
    def __init__(self, possible_latent, priors, models):
        self.z = possible_latent
        self.models = models  # dictionary z: model[z]
        self.priors = priors  # dictionary z: prior[z]

    def __e_step(self, observed_values):
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

        return qs

    def __m_step(self, qs, observed_values, thr=0.001):
        new_priors = {}
        for z in self.z:
            p = np.mean(qs[z])
            # p = np.fmax(thr, p)
            # p = np.fmin(1 - thr, p)
            new_priors[z] = p

        new_theta = {}
        for z in self.z:
            new_theta[z] = self.models[z].argmax_for_parameters(qs[z], observed_values)

        return new_priors, new_theta

    def check_convergence(self, old_parameters, new_parameters, tol):
        new_unmatched, new_matched, new_prior0, new_prior1 = new_parameters
        old_unmatched, old_matched, old_prior0, old_prior1 = old_parameters

        # compare priors
        for o, n in zip([old_prior0, old_prior1], [new_prior0, new_prior1]):
            if np.abs(o-n) > tol:
                return False

        def flatten(l):
            flat_list = [item for sublist in l for item in sublist]
            return np.array(flat_list)

        new_unmatched = flatten(new_unmatched)
        new_matched = flatten(new_matched)
        old_unmatched = flatten(old_unmatched)
        old_matched = flatten(old_matched)

        for o, n in zip([old_matched, old_unmatched], [new_matched, new_unmatched]):
            if not np.allclose(o, n, tol):
                return False

        return True

    def optimize(self, observed_values, max_step, tolerance):
        # TODO reasonable parameter recording
        old_parameters = [self.models[z].parameters for z in self.z]
        old_parameters.extend([self.priors[z] for z in self.z])

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

            print()

            # recorded_parameters.append(old_parameters)
            old_parameters = new_parameters

        return self.models # recorded_parameters