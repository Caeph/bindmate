import time
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm
import tensorflow as tf


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def strictly_positive_constraint(x):
    # Set the minimum value you consider "strictly positive"
    # You can adjust this value based on your needs
    min_value = 1e-6  # Example: One millionth
    return tf.maximum(x, min_value)


def strictly_between_one_and_zero(x):
    min_value = 1e-6  # Example: One millionth
    val = tf.maximum(x, min_value)
    val = tf.minimum(val, 1-min_value)
    return val


class MStepGMMOptimizer:

    def __constraint_penalties(self, variables):
        # this is a stub - this will be a non-linear function depending on observed_values and qs

        # x is a list of tf.Variable(loc, scale, weight)
        # include a constraint to ensure that the sum of weights over all variables equals to one.
        penalty = 0
        for z in self.zs:
            for m in self.ms:
                weights = tf.stack([variables[i] for i in self.about_variables[z][m] if i % 3 == 2])

                weights = tf.stack(weights)
                weight_constraint = tf.reduce_sum(weights) - 1.0  # Constraint: sum of weights - 1 should be 0

                # Apply a penalty for violating the weight constraint
                penalty += self.penalty_factor * tf.square(
                    weight_constraint)  # Squared penalty for smoother gradient descent

                # weights should also be positive
                positivity_constraint = tf.reduce_sum(tf.nn.relu(-weights))
                penalty += self.penalty_factor * tf.square(
                    positivity_constraint)
        return penalty

    def gaussian_pdf(self, x, mean, std):
        """Calculate the Gaussian probability density function."""
        result = tf.exp(-0.5 * tf.square((x - mean) / std)) / (std * tf.sqrt(2.0 * np.pi))

        return result

    def gaussian_mixture_probabilities(self, m_observed_values, m_z_parameters):
        total = tf.zeros_like(m_observed_values)

        for var in chunks(m_z_parameters, 3):
            loc, scale, weight = var[0], var[1], var[2]
            addition = tf.nn.relu(weight) * self.gaussian_pdf(m_observed_values, loc, scale)
            # if tf.reduce_any(tf.math.is_nan(addition)):
            #     print("NAN")
            total = total + addition
        total = tf.nn.relu(total)
        return total

    def gaussian_mixture_probabilities_z(self, variables, z):
        m_z_probas = []
        for m in self.ms:
            m_z_params = [variables[i] for i in self.about_variables[z][m]]
            m_z_probas.append(self.gaussian_mixture_probabilities(self.observed_values[:, m], m_z_params))

        z_probas = tf.stack(
            m_z_probas
        )
        return z_probas

    def feature_weight(self, probabilities, pseudocount=1e-8):
        probabilities_uncond = tf.reduce_sum(probabilities, axis=1)  # m x NO
        rep_probabilities_uncond = tf.tile(tf.expand_dims(probabilities_uncond, 1), [1, probabilities.shape[1], 1])
        rep_priors = tf.tile(tf.reshape(self.priors, [1, 3, 1]),
                             [probabilities.shape[0], 1, probabilities.shape[-1]])

        numerator = tf.reduce_sum(
            probabilities * rep_priors * -tf.nn.relu(
                -tf.math.log((probabilities / (rep_probabilities_uncond + pseudocount)) + pseudocount)), axis=-1

        )
        denominator = tf.reduce_sum(
            probabilities_uncond * -tf.nn.relu(-tf.math.log(probabilities_uncond + pseudocount)), axis=-1)
        denominator = tf.tile(tf.expand_dims(denominator, 1), [1, numerator.shape[1]])

        feature_weights = tf.nn.relu(numerator / (denominator + pseudocount))

        Z = tf.reduce_sum(feature_weights) + pseudocount
        Z = Z / (len(self.ms) * len(self.zs))

        feature_weights = feature_weights / Z

        return feature_weights

    def __init__(self, initial_parameter_values, observed_values, qs, new_priors, learning_rate=0.2,
                 penalty_factor=10000):
        # initial parameter values is a 2d dict of list of triples (loc, scale, weight) for each m and z
        # self.variables = [tf.Variable([gmm_theta[0], gmm_theta[1], gmm_theta[2]],
        #                               dtype=tf.float32) for gmm_theta in initial_parameter_values]
        self.variables = {}

        self.zs = []
        self.ms = []
        model_theta = []
        model_theta_about = {}
        global_index_counter = 0
        for z in initial_parameter_values:
            self.zs.append(z)
            model_theta_about[z] = {}
            for m in initial_parameter_values[z]:
                self.ms.append(m)
                if m not in model_theta_about[z]:
                    model_theta_about[z][m] = []
                for i, params in enumerate(initial_parameter_values[z][m]):
                    theta = [
                        tf.Variable(initial_value=[params[0]], dtype=tf.float32, name=f"loc:{i}:{m}:{z}"),  # loc
                        tf.Variable(initial_value=[params[1]],
                                    dtype=tf.float32,
                                    name=f"scale:{i}:{m}:{z}",
                                    constraint=strictly_positive_constraint
                                    ),  # scale
                        tf.Variable(initial_value=[params[2]],  # Example initial values
                                    dtype=tf.float32,
                                    name=f"model weight:{i}:{m}:{z}",
                                    constraint=strictly_between_one_and_zero)  # model weight
                    ]
                    model_theta.extend(theta)

                    model_theta_about[z][m].append(global_index_counter)
                    model_theta_about[z][m].append(global_index_counter + 1)
                    model_theta_about[z][m].append(global_index_counter + 2)
                    global_index_counter += 3
        self.variables = model_theta
        self.about_variables = model_theta_about

        self.ms = list(set(self.ms))  # unique

        # Transform a np 2d matrix to tensor, constant
        # shape: (no_examples, no_metrics)
        self.observed_values = tf.constant(observed_values, dtype=tf.float32)

        # Transform a dict [0..len(qs)] -> np array to a tensor matrix, constant
        # shape: (no_examples, no_z)
        self.qs = tf.constant(np.array([qs[z] for z in sorted(qs.keys())]), dtype=tf.float32)
        self.priors = tf.constant(new_priors, dtype=tf.float32)

        # self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

        # Set up the learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1000,
            decay_rate=0.95
        )

        # Create an optimizer with the learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # minimizing func

    def function_to_optimize(self, variables, pseudocount=1e-10):
        penalty = self.__constraint_penalties(variables)

        # Loss function calculation:
        # sum_z sum_m sum_i=1^n qs[z]_i . w(m,z) . log Pr[x_i^m | z],
        # where Pr[x_i^m | z] follows a Gaussian mixture model
        # shape of probabilities -- (z,m,no_points)

        probabilities = tf.stack([self.gaussian_mixture_probabilities_z(variables, z) for z in self.zs])
        probabilities = tf.transpose(probabilities, perm=[1, 0, 2])
        # if tf.reduce_any(tf.math.is_nan(probabilities)):
        #     print("NAN")

        multiples = [len(self.ms)] + [1 for _ in range(tf.shape(self.qs).numpy().size)]
        rep_qs = tf.tile(tf.expand_dims(self.qs, 0), multiples)

        original_loss = tf.reduce_sum(rep_qs * tf.math.log(probabilities + pseudocount), axis=-1)  # get a m x z matrix
        # if tf.reduce_any(tf.math.is_nan(original_loss)):
        #     print("NAN")

        feature_weights = self.feature_weight(probabilities)
        original_loss = - tf.reduce_sum(original_loss * feature_weights)
        # if tf.reduce_any(tf.math.is_nan(original_loss)):
        #     print("NAN")

        return original_loss + penalty

    def optimize(self, ident, iterations=1000, # tol=0.1, staying_thr=50
                 ):
        """
        Performs the optimization of the arbitrary function.
        :param iterations: Number of iterations for the optimization process.
        """
        last_seen_loss = np.inf
        staying = 0

        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Record the operations for automatic differentiation
                tape.watch(self.variables)
                loss = self.function_to_optimize(self.variables)

            # if tf.math.abs(loss - last_seen_loss) < tol:
            #     staying += 1
            # else:
            #     staying = 0
            # last_seen_loss = loss
            # if staying >= staying_thr:
            #     print(f"{ident}:\tCONVERGED loss: {loss.numpy()}")
            #     return loss

            # Compute gradients
            gradients = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(zip(gradients, self.variables))

            # Logging
            if i % 250 == 0:
                print(f"{ident}:\tOptimization iteration {i}, Loss: {loss.numpy()}")
        return loss

    def get_optimized_parameters(self):
        """
        Returns the optimized parameters.
        """
        best_params = {}
        for z in self.about_variables:
            best_params[z] = []
            for m in sorted(self.about_variables[z].keys()):
                flattened = [self.variables[i].numpy()[0] for i in self.about_variables[z][m]]
                best_params[z].append(flattened)

        best_probabilities = tf.stack([self.gaussian_mixture_probabilities_z(self.variables, z) for z in self.zs])
        best_probabilities = tf.transpose(best_probabilities, perm=[1, 0, 2])
        best_weights = self.feature_weight(best_probabilities)

        return best_params, best_weights.numpy()


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
            # self.parameters = [random_shift(params['initial_parameters']) for params in optim_info]
            self.parameters = [params['initial_parameters'] for params in optim_info]
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


class ProbabilityModelEnsemble:
    def __init__(self, models, ident="MODEL"):
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

    def __init__(self, models, batch_identificator):
        super().__init__(models)
        self.weights = None
        self.ident = batch_identificator
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

    def argmax_params(self, qs, observed_values, new_priors, gmm_only=True, random_init=True, no_tries=5,
                      optimizer_max_iter=1500, optimizer_tol=0.01, init_params_perturbation_proba=0.1,
                      learning_rate=1, constraint_violation_penalty=10000
                      ):
        init_params = {}

        for z in range(len(self.models)):
            init_params[z] = {}
            for m in range(observed_values.shape[1]):
                perturb = np.random.uniform(0,1)
                if perturb <= init_params_perturbation_proba:
                    init_params[z][m] = list(chunks(self.models[z].parameters[m], 3))
                else:
                    random_subset_indices = np.random.randint(low=0, high=len(observed_values),
                                                              size=int(len(observed_values) * 0.33))
                    n = len(random_subset_indices)
                    chosen = observed_values[random_subset_indices, m]
                    loc_estim = np.mean(chosen)
                    scale_estim = np.sqrt(np.sum((chosen - loc_estim)**2) / (n-1))
                    init_params[z][m] = [[
                        np.quantile(chosen, i / (len(self.models[z].parameters[m]) // 3)),
                        scale_estim + np.random.normal(loc=0, scale=scale_estim/2),
                        np.random.uniform(0,1)
                    ] for i, _ in enumerate(chunks(self.models[z].parameters[m], 3))]

        optimizer = MStepGMMOptimizer(
            initial_parameter_values=init_params,
            observed_values=observed_values,
            qs=qs,
            new_priors=[new_priors[z] for z in sorted(qs.keys())],
            learning_rate=learning_rate,
            penalty_factor=constraint_violation_penalty
        )
        best_achieved_fun_value = optimizer.optimize(self.ident, optimizer_max_iter)
        optimized_params, optimized_weights = optimizer.get_optimized_parameters()

        new_weights = {}
        for z in range(len(self.models)):
            new_weights[z] = {}
            for m in range(observed_values.shape[1]):
                new_weights[z][m] = optimized_weights[m,z]

        return optimized_params, best_achieved_fun_value

        # def all_probas_for_m(params, m, pseudocount=1e-10):
        #     # xmi calculation
        #     m = int(m)
        #     x = pd.Series(observed_values[:, m])
        #     p_xmi = np.zeros((len(new_priors), len(observed_values)))
        #     for z in self.models.keys():
        #         model = self.models[z]
        #         param = params[z]  # [m]
        #         p_func = model.optimization_info[m]['proba']
        #         p_xmi[z, :] += x.swifter.progress_bar(False).apply(lambda xi: p_func(xi, param) + pseudocount)
        #     return p_xmi
        #
        # def KL_divergence_weight(z, p_xmi, pseudocount=1e-9):
        #     p_xi = np.sum(p_xmi[:, :], axis=0) + pseudocount
        #     p_xi_given_mz = p_xmi[z, :]
        #
        #     w_avg = np.sum((p_xi_given_mz * new_priors[z]) * np.log(
        #         (p_xi_given_mz / (p_xi + pseudocount))
        #     ))
        #     bottom_entropy = np.sum(p_xi * np.log(p_xi)) + pseudocount
        #
        #     return w_avg / bottom_entropy
        #
        # def calculate_weights_given_m(p_xmi, minimal_weight=None):
        #     weights = {}
        #     for z in self.models.keys():
        #         for m in self.weights[z].keys():
        #             weight_mz = KL_divergence_weight(z, p_xmi)
        #             weights[z] = weight_mz
        #
        #     # normalize weights
        #     Z = np.sum([weights[z] for z in self.models.keys()])
        #     no_z = len(self.models.keys())
        #     # no_m = observed_values.shape[1]
        #     # Z = (Z / (no_z * no_m))
        #     Z = (Z / no_z)
        #     for z in self.models.keys():
        #         weights[z] = weights[z] / Z
        #         if minimal_weight is not None:
        #             weights[z] = np.fmax(weights[z], minimal_weight)
        #     return weights
        #
        # def objective_for_m(params, pseudocount=1e-6):
        #     # read params
        #     m = params[0]
        #     if m >= observed_values.shape[1]:
        #         return np.inf
        #     params = flat_array_to_m_params_dict(params[1:])
        #
        #     p_xmi = all_probas_for_m(params, m)
        #     weights = calculate_weights_given_m(p_xmi)
        #
        #     # do calculation
        #     result = 0
        #     for z in self.models.keys():
        #         p = p_xmi[z, :]
        #         weight_mz = weights[z]
        #         current_res = weight_mz * np.sum(np.log(p + pseudocount) * qs[z])
        #         result = result + current_res
        #
        #     if np.isnan(result):
        #         return np.inf
        #
        #     return - result
        #
        # total_objective_values = 0
        # best_params = {z: [] for z in range(len(self.models))}
        #
        # new_weights = {z: dict() for z in range(len(self.models))}
        # for m in range(observed_values.shape[1]):
        #     # get params
        #     init_params = [m]
        #     about_params = []
        #     param_bounds = [[m, m]]
        #
        #     for z in range(len(self.models)):
        #         init_params.extend(self.models[z].parameters[m])
        #         param_bounds.extend([check_bounds(bounds, observed_values[:, m]) for bounds in
        #                              self.models[z].optimization_info[m]['params_bounds']])
        #         about_params.append(len(self.models[z].parameters[m]))
        #
        #     def flat_array_to_m_params_dict(params_vector):
        #         index = 0
        #         params = {}
        #         about_i = 0
        #         for z in range(len(self.models)):
        #             l = about_params[about_i]
        #             params[z] = params_vector[index:index + l]
        #             about_i += 1
        #             index += l
        #         return params
        #
        #     # optimize objective
        #     objective_for_m(init_params)
        #     init_params = np.array(init_params)
        #
        #     def chunks(lst, n):
        #         """Yield successive n-sized chunks from lst."""
        #         for i in range(0, len(lst), n):
        #             yield lst[i:i + n]
        #
        #     if gmm_only:
        #         # def weight_sum_to_one(params, z):
        #         #     params = flat_array_to_m_params_dict(params[1:])[z]
        #         #     w_sum = np.sum([params[i*3+2] for i in range(len(params) // 3)])
        #         #     return 1 - w_sum
        #         #
        #         # constraints = ({"type": 'ineq',
        #         #                 'fun': lambda params: weight_sum_to_one(params, z)} for z in range(len(self.models)))
        #         #
        #         # def all_z_weights_sum(params):
        #         #     return np.array([weight_sum_to_one(params, z) for z in range(len(self.models))])
        #         weight_params = [i*3 for i in range(1, (len(init_params) // 3)+1)]
        #         no_models = len(weight_params) / len(self.models)
        #         weight_indices = [wi_array for wi_array in chunks(weight_params, int(no_models))]
        #
        #         start = time.time()
        #
        #         A = []
        #         for wa in weight_indices:
        #             arr = np.zeros_like(init_params)
        #             for i in wa:
        #                 arr[i] = 1
        #             A.append(arr)
        #
        #         A = np.stack(A)
        #         lb = np.ones(len(self.models)) - 0.05
        #         ub = np.ones(len(self.models))
        #         constraints = optimize.LinearConstraint(A, lb, ub)
        #
        #         minimizers = []
        #         minimizing = optimize.minimize(
        #             objective_for_m,
        #             init_params,
        #             bounds=param_bounds,
        #             constraints=constraints,
        #             method='COBYLA', options={'tol': optimizer_tol,
        #                                       'maxiter': optimizer_max_iter,
        #                                       "time_limit": optimizer_time_limit})
        #         minimizers.append(minimizing)
        #
        #         for i_trial in range(no_tries):
        #             ith_init_params = []
        #             for lower, upper in param_bounds:
        #                 ith_init_params.append(np.random.uniform(lower, upper))
        #
        #             ith_init_params = np.array(ith_init_params)
        #             minimizing = optimize.minimize(
        #                 objective_for_m,
        #                 ith_init_params,
        #                 bounds=param_bounds,
        #                 constraints=constraints,
        #                 method='COBYLA', options={'tol': optimizer_tol,
        #                                           'maxiter': optimizer_max_iter,
        #                                           "time_limit": optimizer_time_limit})
        #             minimizers.append(minimizing)
        #
        #         print(f"Time spent in minimizing: {time.time() - start}")
        #
        #         valid = [m.fun for m in minimizers if m.success]
        #         if len(valid) > 0:
        #             chosen = np.argmin(valid)
        #         else:
        #             chosen = np.argmin([m.fun for m in minimizers])
        #         minimizing = minimizers[chosen]
        #     else:
        #         minimizing = optimize.minimize(objective_for_m,
        #                                        init_params,
        #                                        bounds=param_bounds,
        #                                        method='SLSQP')
        #
        #     total_objective_values = total_objective_values + minimizing.fun
        #     m_best_params_dict = flat_array_to_m_params_dict(minimizing.x[1:])  # first parameter is m
        #
        #     # parameters checkup - weights are within bounds, ...
        #     # sort params
        #
        #     for z in m_best_params_dict.keys():
        #         checked_params = []
        #         locs = []
        #         sum_of_weights = sum([weight if weight > 0 else 1e-6 for _, _, weight in
        #                               chunks(m_best_params_dict[z], 3)])
        #
        #         for loc, scale, weight in chunks(m_best_params_dict[z], 3):
        #             if weight < 0:
        #                 weight = 1e-6
        #
        #             checked_params.append([loc, scale, weight / sum_of_weights])
        #             locs.append(loc)
        #         checked_params = np.array(checked_params)[np.argsort(locs)]
        #         m_best_params_dict[z] = [y for x in checked_params for y in x]  # flattened
        #     #
        #
        #     best_p_xmi = all_probas_for_m(m_best_params_dict, m)
        #     best_weights_m = calculate_weights_given_m(best_p_xmi, minimal_weight=1e-6)
        #     for z in range(len(self.models)):
        #         best_params[z].append(m_best_params_dict[z])
        #         new_weights[z][m] = best_weights_m[z]
        #
        # self.weights = new_weights
        # return best_params, total_objective_values

        # init_params = []
        # about_params = []
        # param_bounds = []
        # for m in range(observed_values.shape[1]):
        #     for z in range(len(self.models)):
        #         init_params = []
        #         about_params = []
        #         param_bounds = []
        #         #
        #         if init_guess == 'previous':
        #             init_params.extend(self.models[z].parameters[m])
        #         elif init_guess == 'random':
        #             init_params.extend(self.models[z].parameters[m])
        #         else:
        #             raise NotImplementedError("Unknown method of getting initial guess.")
        #         param_bounds.extend([check_bounds(bounds, observed_values[:, m]) for bounds in self.models[z].optimization_info[m]['params_bounds']])
        #         about_params.append(len(self.models[z].parameters[m]))

        # def flat_array_to_params_dict(params_vector):
        #     index = 0
        #     params = {}
        #     about_i = 0
        #     for z in range(len(self.models)):
        #         params[z] = []
        #         for m in range(observed_values.shape[1]):
        #             l = about_params[about_i]
        #             params[z].append(params_vector[index:index+l])
        #             about_i += 1
        #             index += l
        #     return params
        #
        # # numerically minimize objective function
        #
        #
        # # TODO this might need speeding up
        # minimizing = optimize.minimize(full_objective, np.array(init_params), bounds=param_bounds, method='SLSQP'
        #                                )
        # print(minimizing.message)
        # # minimizing = optimize.dual_annealing(full_objective, bounds=param_bounds)
        # best_params = flat_array_to_params_dict(minimizing.x)
        #
        # best_p_xmi = all_probas(best_params)
        # self.weights = calculate_weights(best_p_xmi, minimal_weight=1e-6)
        # # print(f"Achieved best weights: {self.weights}")
        #
        # return best_params, minimizing.fun


# class GmmWeightedModelEnsemble(WeightedModelEnsemble):
#     def __init__(self, models, max_no_models=10):
#         super().__init__(models)
#         self.max_no_models = max_no_models
# ALL Pr[x_im | z] is calculated as a linear combination of several Gaussians -- set up for every one


def numerical_argmax_func(observed_vector, model_qs, probafunc, param_bounds):
    def objective(params):
        a = model_qs * np.log(probafunc(observed_vector, params) + 1e-6)

        return -np.sum(a[~np.isnan(a)])

    minimizing = optimize.dual_annealing(objective, param_bounds)
    return minimizing


def check_bounds(b, observed):
    if b is None:
        return 0 + 10e-6, max(observed)

    lower, upper = b
    if upper == 1:
        return 0 + 1e-6, 1

    lower = 1e-6
    upper = max(observed)
    return lower, upper


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
            param = self.parameters[i]
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
        # print(f"running argmax for params in weighted proba model {self.identificator}")

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
    def __init__(self, possible_latent, priors, models,
                 batch_identificator,
                 m_step_implementation='weighted'):
        self.z = possible_latent
        # self.models = models  # dictionary z: model[z]
        if m_step_implementation == 'weighted':
            self.models = WeightedModelEnsemble(models, batch_identificator)
        elif m_step_implementation == 'simple':
            self.models = ProbabilityModelEnsemble(models)
        # elif m_step_implementation == 'gmm_weighted':
        #     self.models = GmmWeightedModelEnsemble(models)
        #     # TODO
        else:
            raise NotImplementedError(m_step_implementation)

        self.priors = priors  # dictionary z: prior[z]
        self.m_step_implementation_type = m_step_implementation
        self.current_objective_value = np.inf
        self.pseudocount = 1e-8

    def __e_step(self, observed_values):
        # print("E-step started")
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
            qs[z] = qs[z] / (bottom + self.pseudocount)

        # print("E-step done")
        return qs

    def __m_step(self, qs, observed_values, thr=1e-50):
        # print("M-step started")
        new_priors = {}
        total = 0
        for z in self.z:
            p = np.mean(qs[z])
            p = np.fmax(thr, p)
            p = np.fmin(1 - thr, p)
            total += p
            new_priors[z] = p
        new_priors = {x: new_priors[x] / total for x in new_priors}

        # new_theta = {}
        # for z in self.z:
        #     new_theta[z] = self.models[z].argmax_for_parameters(qs[z], observed_values)
        new_theta, objective_value = self.models.argmax_params(qs, observed_values, new_priors)

        # print("M-step done")
        return new_priors, new_theta, objective_value

    def __record(self, parameters, colector, sep=';', subsep=';'):
        # TODO
        if colector is None:
            return
        #
        # def flatten_to_string(l):
        #     flat_list = [str(item) for sublist in l for item in sublist]
        #     return np.array(flat_list)
        #
        # # unmatched, matched, prior0, prior1 = parameters
        # # unmatched = subsep.join(flatten_to_string(unmatched))
        # # matched = subsep.join(flatten_to_string(matched))
        # # print(sep.join([unmatched, matched, str(prior0), str(prior1)]), file=colector)
        # priors = [str(x) for x in parameters[-len(self.models):]]
        # model_params = [subsep.join(flatten_to_string(x)) for x in parameters[:-len(self.models)]]
        # print(sep.join([*model_params, *priors]), file=colector)

    def check_convergence(self, old_parameters, new_parameters, objective_value, tol):
        if np.abs(objective_value - self.current_objective_value) <= tol:
            return True
        self.current_objective_value = objective_value
        return False

        # def flatten(l):
        #     flat_list = [item for sublist in l for item in sublist]
        #     return np.array(flat_list)
        #
        # new_params, new_priors = new_parameters[:-len(self.models)], new_parameters[-len(self.models):]
        # old_params, old_priors = old_parameters[:-len(self.models)], old_parameters[-len(self.models):]
        #
        # # compare priors:
        # for o, n in zip(old_priors, new_priors):
        #     if np.abs(o - n) > tol:
        #         return False
        #
        # # compare models
        # for o, n in zip(old_params, new_params):
        #     if not np.allclose(flatten(o), flatten(n), tol):
        #         return False

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

    def optimize(self, observed_values, max_step, tolerance,
                 parameter_colector=None, identificator=""):
        old_parameters = [self.models.get_models()[z].parameters for z in self.z]
        old_parameters.extend([self.priors[z] for z in self.z])
        self.__record(old_parameters, colector=parameter_colector)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(max_step):
                qs = self.__e_step(observed_values)
                new_priors, new_theta, objective_value = self.__m_step(qs, observed_values)

                print(f"{identificator}: ITERATION {i}")
                print(f"{identificator}: objective func value: {objective_value}")
                for z in self.priors:
                    print(f"{identificator}\tprior({z}): {self.priors[z]} --> {new_priors[z]}\t")
                for z in self.z:
                    old_params = self.models.get_models()[z].parameters
                    for m, old_m_params in enumerate(old_params):
                        o = ', '.join(["%.2f" % x for x in old_m_params])
                        n = ', '.join(["%.2f" % x for x in new_theta[z][m]])
                        print(f"{identificator}\ttheta(z={z}, m={m}): {o} --> {n}\t")

                self.priors = new_priors
                for z in self.z:
                    # self.models[z].set_parameters(new_theta[z])
                    self.models.set_params(new_theta[z], z)

                new_parameters = [self.models.get_models()[z].parameters for z in self.z]
                new_parameters.extend([self.priors[z] for z in self.z])
                # print(f"{identificator}: OLD: {old_parameters}")
                # print(f"{identificator}: NEW: {new_parameters}")

                convergence = self.check_convergence(old_parameters, new_parameters, objective_value, tolerance)
                if convergence:
                    print("CONVERGENCE")
                    break

                # recorded_parameters.append(old_parameters)
                self.__record(new_parameters, colector=parameter_colector)
                old_parameters = new_parameters

        return self.models.get_models()  # recorded_parameters
