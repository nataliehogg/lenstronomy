__author__ = "nataliehogg"

import jax 
import jax.numpy as jnp 
import blackjax 
import blackjax.ns.adaptive

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler


class BlackJaxNestedSampler(NestedSampler):
        '''
        Wrapper for the nested sampler in Blackjax.
        '''

    def __init__(
        self, 
        likelihood_module,
        prior_type="uniform",
        prior_means=None,
        prior_sigmas=None,
        width_scale=1,
        sigma_scale=1,
        mpi=False,
        **kwargs
        ):


        self._likelihood_module = likelihood_module

    def run(self):
        '''
        Run the nested sampler.
        '''

        n_live = 100
        n_delete = 20
        num_mcmc_steps = 100 #d * 5
        ndead_max = 10000
        
        algo = blackjax.ns.adaptive.nss(
            logprior_fn=self.prior, #logprior_fn, 
            loglikelihood_fn=self.log_likelihood # calls the logL from base_nested_sampler (?)
            n_delete=n_delete,
            num_mcmc_steps=num_mcmc_steps,
            )


        # sample points from the prior
        rng_key, init_key = jax.random.split(rng_key, 2)
        
        initial_particles = jax.random.multivariate_normal(
            init_key, 
            prior_mean, 
            prior_cov, 
            shape=(n_live,)
            )
            
        state = algo.init(initial_particles, loglikelihood_fn)

        # run the nested sampler

        while not state.sampler_state.logZ_live - state.sampler_state.logZ < -5:
            (state, rng_key), dead_info = self.one_step((state, rng_key), None)
            dead.append(dead_info)



    @jax.jit
    def one_step(self, carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

# def likelihood_for_cobaya(**kwargs):
#     """We define a function to return the log-likelihood; this function is
#     passed to Cobaya. The function must be nested within the run() function for
#     it to work properly.

#     :param kwargs: dictionary of keyword arguments
#     """
#     current_input_values = [kwargs[p] for p in sampled_params]
#     logp = self._likelihood_module.likelihood(current_input_values)
#     return logp