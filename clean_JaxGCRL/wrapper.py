import jax
import jax.numpy as jnp

from flax import struct
from brax import envs
from brax.envs.base import State, Dict, Wrapper


@struct.dataclass
class EvalMetrics:
    """Dataclass holding evaluation metrics for Brax.

    Attributes:
        episode_metrics: Aggregated episode metrics since the beginning of the
          episode.
        active_episodes: Boolean vector tracking which episodes are not done yet.
        episode_steps: Integer vector tracking the number of steps in the episode.
    """

    episode_sum_metrics: Dict[str, jax.Array]
    episode_metrics: Dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array


class EvalWrapper(Wrapper):
    """Brax env with eval metrics."""

    def reset(self, rng) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        # tmp_dict = reset_state.metrics.copy()
        # tmp_dict.pop('coverage_map', None)

        sum_metrics = reset_state.metrics.copy()
        position = sum_metrics.pop('position', None)
        waypoint_position = sum_metrics.pop('waypoint_position', None)
        target_position = sum_metrics.pop('target_position', None)
        velocity = sum_metrics.pop('velocity', None)
        metrics = dict(
            position=-jnp.ones((position.shape[0], self.env.episode_length, position.shape[1])),
            waypoint_position=-jnp.ones((waypoint_position.shape[0], self.env.episode_length,
                                         waypoint_position.shape[1])),
            target_position=-jnp.ones((target_position.shape[0], self.env.episode_length,
                                       target_position.shape[1])),
            velocity=-jnp.ones((velocity.shape[0], self.env.episode_length, velocity.shape[1])),
        )

        eval_metrics = EvalMetrics(
            episode_sum_metrics=jax.tree_util.tree_map(jnp.zeros_like,  sum_metrics),
            episode_metrics=jax.tree_util.tree_map(lambda x: -jnp.ones_like(x) * jnp.inf, metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward, dtype=jnp.int32),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info['eval_metrics']

        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f'Incorrect type for state_metrics: {type(state_metrics)}')
        del state.info['eval_metrics']

        nstate = self.env.step(state, action)
        nstate.metrics['reward'] = nstate.reward

        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info['steps'].astype(jnp.int32),
            state_metrics.episode_steps,
        )

        sum_metrics = nstate.metrics.copy()
        metrics = dict(
            position=sum_metrics.pop('position', None),
            waypoint_position=sum_metrics.pop('waypoint_position', None),
            target_position=sum_metrics.pop('target_position', None),
            velocity=sum_metrics.pop('velocity', None),
        )

        episode_sum_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_sum_metrics,
            sum_metrics,
        )
        # episode_metrics = jax.tree_util.tree_map(
        #     lambda a, b: a + b * state_metrics.active_episodes,
        #     state_metrics.episode_metrics,
        #     metrics,
        # )

        # def true_f(metric):
        #     # x = jax.tree_util.tree_map(
        #     #     lambda a, b: a.at[jnp.arange(a.shape[0]), state_metrics.episode_steps].set(b),
        #     #     state_metrics.episode_metrics,
        #     #     metrics
        #     # )
        #
        #     state_metrics.episode_metrics['x_position'] = state_metrics.episode_metrics['x_position'].at[
        #         jnp.arange(state_metrics.episode_metrics['x_position'].shape[0]), state_metrics.episode_steps].set(
        #         metrics['x_position']
        #     )
        #     x_pos = state_metrics.episode_metrics['x_position']
        #
        #     return x_pos
        #
        # def false_f():
        #     return state_metrics.episode_metrics['x_position']

        # def update_f(active_episode, episode_step, ):
        metrics = jax.tree_util.tree_map(
            lambda a: jnp.where(state_metrics.active_episodes[:, None], a, -jnp.inf),
            metrics
        )

        # def update(a, b):
        #     a = a.at[jnp.arange(a.shape[0]), state_metrics.episode_steps].set(b)
        #
        #     return a

        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a.at[jnp.arange(a.shape[0]), state_metrics.episode_steps].set(b),
            state_metrics.episode_metrics,
            metrics
        )

        # state_metrics.episode_metrics['x_position'] = state_metrics.episode_metrics['x_position'].at[
        #     jnp.arange(state_metrics.episode_metrics['x_position'].shape[0]), state_metrics.episode_steps].set(
        #     metrics['x_position']
        # )
        # x_pos = state_metrics.episode_metrics['x_position']

        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = EvalMetrics(
            episode_sum_metrics=episode_sum_metrics,
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )

        nstate.info['eval_metrics'] = eval_metrics
        return nstate
