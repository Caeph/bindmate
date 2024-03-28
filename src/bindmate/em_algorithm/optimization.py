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
    val = tf.minimum(val, 1 - min_value)
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

    def optimize(self, ident, iterations=1000,  # tol=0.1, staying_thr=50
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
                      optimizer_max_iter=100, optimizer_tol=0.01, init_params_perturbation_proba=0.1,
                      learning_rate=1, constraint_violation_penalty=10000
                      ):
        init_params = {}

        for z in range(len(self.models)):
            init_params[z] = {}
            for m in range(observed_values.shape[1]):
                perturb = np.random.uniform(0, 1)
                if perturb <= init_params_perturbation_proba:
                    init_params[z][m] = list(chunks(self.models[z].parameters[m], 3))
                else:
                    random_subset_indices = np.random.randint(low=0, high=len(observed_values),
                                                              size=int(len(observed_values) * 0.33))
                    n = len(random_subset_indices)
                    chosen = observed_values[random_subset_indices, m]
                    loc_estim = np.mean(chosen)
                    scale_estim = np.sqrt(np.sum((chosen - loc_estim) ** 2) / (n - 1))
                    init_params[z][m] = [[
                        np.quantile(chosen, i / (len(self.models[z].parameters[m]) // 3)),
                        scale_estim + np.random.normal(loc=0, scale=scale_estim / 2),
                        np.random.uniform(0, 1)
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
                new_weights[z][m] = optimized_weights[m, z]

        return optimized_params, best_achieved_fun_value


def strictly_between_one_and_zero_and_norm_to_one(x):
    min_value = 1e-6  # Example: One millionth
    val = tf.maximum(x, min_value)

    return val


class MStepGMMOptimizerSingleM:
    # both m and z is given

    def initialize_params(self, no_gmm_models, observed_values):
        variables = []
        for i in range(no_gmm_models):
            # set up a gmm parameters model
            chosen = np.random.choice(observed_values, size=int((1 / no_gmm_models) * len(observed_values)))
            init_loc, init_scale = (np.quantile(chosen, i / no_gmm_models),
                                    np.random.normal(10, 3))  # TODO
            variables.append(
                tf.Variable(initial_value=[init_loc], dtype=tf.float32, name=f"loc:{i}")
            )
            variables.append(
                tf.Variable(initial_value=[init_scale],
                            dtype=tf.float32,
                            name=f"scale:{i}",
                            constraint=strictly_positive_constraint
                            )
            )
        init_weight = np.array([np.random.randint(low=1, high=10) for _ in range(no_gmm_models)])
        init_weight = init_weight / init_weight.sum()
        variables.append(
            tf.Variable(
                initial_value=init_weight,
                dtype=tf.float32,
                name="weight_vector",
                constraint=strictly_between_one_and_zero_and_norm_to_one
            )
        )
        return variables

    def __init__(self, observed_values, qs_z, new_z_prior, no_gmm_models, learning_rate=0.1,
                 penalty_factor=10000, max_iter=1000):
        variables = self.initialize_params(no_gmm_models, observed_values)
        self.variables = variables

        # Transform a np 1d matrix to tensor, constant
        # shape: (no_examples)
        self.observed_values = tf.constant(observed_values, dtype=tf.float32)
        values, counts = np.unique(observed_values, return_counts=True)
        x_mi_probabilities = {v: (c / len(observed_values)) for v, c in zip(values, counts)}
        self.xmi_probabilities = tf.constant(
            list(map(x_mi_probabilities.get, observed_values)),
            dtype=tf.float32
        )
        self.bottom_entropy = tf.reduce_sum(self.xmi_probabilities * tf.math.log(self.xmi_probabilities))

        # Transform a dict [0..len(qs)] -> np array to a tensor matrix, constant
        # shape: (no_examples, no_z)
        self.qs = tf.constant(qs_z, dtype=tf.float32)
        self.prior = new_z_prior

        # self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

        # Set up the learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=max_iter,
            decay_rate=0.95
        )
        self.max_iter = max_iter

        # Create an optimizer with the learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def __gaussian_pdf(self, x, mean, std):
        """Calculate the Gaussian probability density function."""
        result = tf.exp(-0.5 * tf.square((x - mean) / std)) / (std * tf.sqrt(2.0 * np.pi))
        return result

    def gaussian_mixture_probabilities(self, variables):
        total = tf.zeros_like(self.observed_values)
        gmm_i = 0
        model_weights = variables[-1] / tf.reduce_sum(variables[-1])
        for var in chunks(variables[:-1], 2):
            loc, scale = var[0], var[1]
            weight = model_weights[gmm_i]
            addition = weight * self.__gaussian_pdf(self.observed_values, loc, scale)
            total = total + addition

            gmm_i += 1
        return total

    def feature_weight(self, xmi_probabilities_given_z, pseudocount=1e-8):
        top = xmi_probabilities_given_z * self.prior * tf.math.log(
            (xmi_probabilities_given_z / self.xmi_probabilities) + pseudocount)
        feature_weight = tf.reduce_sum(top) / self.bottom_entropy
        return feature_weight

    def function_to_optimize(self, variables, pseudocount=1e-10):
        # penalty = self.__constraint_penalties(variables)

        probabilities = self.gaussian_mixture_probabilities(variables)
        to_minimize = self.qs * tf.math.log(probabilities + pseudocount)  # 1D

        feature_weight = self.feature_weight(probabilities)
        to_minimize = - tf.reduce_sum(to_minimize * feature_weight)

        return to_minimize

    def optimize(self, ident, tolerance_thr=1e-3, log_check=100
                 ):
        """
        Performs the optimization of the arbitrary function.
        :param iterations: Number of iterations for the optimization process.
        """
        last_loss_seen = np.inf
        for i in range(self.max_iter):
            with tf.GradientTape() as tape:
                # Record the operations for automatic differentiation
                tape.watch(self.variables)
                loss = self.function_to_optimize(self.variables)

            if tf.math.is_nan(loss):
                print(f"{ident}:\tloss is nan, trying a different init")
                self.variables = self.initialize_params(len(self.variables) // 3, self.observed_values)

            # Compute gradients
            gradients = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(zip(gradients, self.variables))

            # Logging
            if i % log_check == 0:
                print(f"{ident}:\tOptimization iteration {i}, Loss: {loss.numpy()}")
                if tf.abs(loss - last_loss_seen) < tolerance_thr:
                    print(f"{ident}\tunchanged from last check, returning...")
                    break
                last_loss_seen = loss
        return loss

    def get_optimized_parameters(self):
        """
        Returns the optimized parameters.
        """
        best_params = self.variables
        best_probas = self.gaussian_mixture_probabilities(best_params)
        feature_weight = self.feature_weight(best_probas)

        gmm_params = best_params[:-1]
        gmm_weights = best_params[-1] / tf.reduce_sum(best_params[-1])
        return gmm_params, gmm_weights, feature_weight.numpy()


class WeightedGMMEnsemble(WeightedModelEnsemble):
    def argmax_params(self, qs, observed_values, new_priors, gmm_only=True, random_init=True, no_tries=5,
                      optimizer_max_iter=500, optimizer_tol=0.01, init_params_perturbation_proba=0.1,
                      learning_rate=1, constraint_violation_penalty=10000
                      ):
        total_achieved_objective = 0
        all_optimized_params = {}
        for z in range(len(qs)):
            all_optimized_params[z] = []
            for m in range(observed_values.shape[1]):
                optimizer = MStepGMMOptimizerSingleM(
                    observed_values=observed_values[:, m],
                    qs_z=qs[z],
                    new_z_prior=new_priors[z],
                    learning_rate=learning_rate,
                    penalty_factor=constraint_violation_penalty,
                    no_gmm_models=len(self.get_models()[z].parameters[m]) // 3,
                    max_iter=optimizer_max_iter
                )
                best_achieved_fun_value = optimizer.optimize(f"{self.ident}-{z}-{m}")
                optimized_params, optimized_model_weights, optimized_weights = optimizer.get_optimized_parameters()
                params_to_send = []
                for w, gm_params in zip(optimized_model_weights, chunks(optimized_params, 2)):
                    loc, scale = gm_params
                    params_to_send.extend([loc.numpy()[0], scale.numpy()[0], w.numpy()])
                all_optimized_params[z].append(params_to_send)
                total_achieved_objective += best_achieved_fun_value.numpy()

        return all_optimized_params, total_achieved_objective


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
            # self.models = WeightedModelEnsemble(models, batch_identificator)
            self.models = WeightedGMMEnsemble(models, batch_identificator)
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
                    for m in range(observed_values.shape[1]):
                        o = ', '.join(["%.2f" % x for x in old_params[m]])
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

        return self.models  # .get_models()  # recorded_parameters
