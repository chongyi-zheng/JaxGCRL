import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import NamedTuple, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
import wandb_osh
from brax import envs
from brax.io import html
from etils import epath
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from wandb_osh.hooks import TriggerWandbSyncHook

from buffer import TrajectoryUniformSamplingQueue
from evaluator import CrlEvaluator
from utils import plot_trajectories


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    log_dir: str = '.'
    wandb_project_name: str = "exploration"
    wandb_entity: str = 'chongyiz1'
    wandb_mode: str = 'offline'
    wandb_group: str = '.'
    capture_video: bool = False
    num_epochs_per_checkpoint: int = 10
    checkpoint: bool = False
    checkpoint_final_rb: bool = False

    # environment specific arguments
    env_id: str = "ant"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 50000000
    num_epochs: int = 50
    num_envs: int = 1024
    num_eval_envs: int = 128
    num_eval_vis: int = 8
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    repr_dim: int = 64
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000

    unroll_length: int = 62

    # to be filled in runtime
    env_steps_per_actor_step: int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps: int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps: int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch: int = 0
    """the number of training steps per epoch(computed in runtime)"""


class SAG_encoder(nn.Module):
    repr_dim: int = 64
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray, g: jnp.ndarray):

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = jnp.concatenate([s, a, g], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.repr_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class SF_encoder(nn.Module):
    repr_dim: int = 64
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, sf: jnp.ndarray):

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(sf)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.repr_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    norm_type: str = "layer_norm"

    log_std_max: int = 2
    log_std_min: int = -5

    @nn.compact
    # def __call__(self, x):
    def __call__(self, s, g_repr):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = jnp.concatenate([s, g_repr], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                    log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))


def render(actor_state, env, exp_dir, exp_name, deterministic=True,
           wandb_track=False):
    def actor_sample(observations, key, deterministic=deterministic):
        means, log_stds = actor.apply(actor_state.params, observations)
        if deterministic:
            actions = nn.tanh(means)
        else:
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

        return actions, {}

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(actor_sample)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    rngs = jax.random.split(rng, 1)
    state = jit_env_reset(rng=rngs)
    for i in range(5000):
        rollout.append(
            jax.tree_util.tree_map(jnp.squeeze, state.pipeline_state))
        act_rng, rng = jax.random.split(rng)
        act_rngs = jax.random.split(act_rng, 1)
        act, _ = jit_inference_fn(state.obs, act_rngs)
        state = jit_env_step(state, act)
        if i % 1000 == 0:
            rngs = jax.random.split(rng, 1)
            state = jit_env_reset(rng=rngs)

    url = html.render(env.sys.replace(dt=env.dt), rollout, height=480)
    with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
        file.write(url)
    if wandb_track:
        wandb.log({"render": wandb.Html(url)})


if __name__ == "__main__":

    args = tyro.cli(Args)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    args.num_training_steps_per_epoch = np.ceil(
        (args.total_env_steps - args.num_prefill_env_steps) / (args.num_epochs * args.env_steps_per_actor_step)
    )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    save_path = os.path.join(args.log_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    if args.track:

        if args.wandb_group == '.':
            args.wandb_group = None

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=save_path,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sag_key, sf_key = jax.random.split(key, 7)

    # Environment setup    
    if args.env_id == "ant":
        from clean_JaxGCRL.envs.ant import Ant

        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    elif "maze" in args.env_id:
        from clean_JaxGCRL.envs.ant_maze import AntMaze

        env = AntMaze(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
            maze_layout_name=args.env_id[4:]
        )

        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    else:
        raise NotImplementedError

    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)

    # Network setup
    # Actor
    actor = Actor(action_size=action_size)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, args.obs_dim]), np.ones([1, args.repr_dim])),
        # params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sag_encoder = SAG_encoder(repr_dim=args.repr_dim)
    sag_encoder_params = sag_encoder.init(
        sag_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]),
        np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    sf_encoder = SF_encoder(repr_dim=args.repr_dim)
    sf_encoder_params = sf_encoder.init(
        sf_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sag_encoder": sag_encoder_params, "sf_encoder": sf_encoder_params},
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Entropy coefficient
    target_entropy = -0.5 * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    actor.apply = jax.jit(actor.apply)
    sag_encoder.apply = jax.jit(sag_encoder.apply)
    sf_encoder.apply = jax.jit(sf_encoder.apply)

    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        extras={
            "state_extras": {
                "truncation": jnp.zeros(()),
                "seed": jnp.zeros(()),
            }
        },
    )


    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer


    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)


    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        """Function to collect data during evaluation. Used in evaluator.py"""

        obs = env_state.obs
        state = obs[:, :args.obs_dim]
        commanded_goal = obs[:, args.obs_dim:]

        commanded_g_repr = sf_encoder.apply(training_state.critic_state.params["sf_encoder"], commanded_goal)

        means, _ = actor_state.apply_fn(training_state.actor_state.params, state, commanded_g_repr)
        # means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )


    def actor_step(training_state, env, env_state, key, extra_fields):
        obs = env_state.obs
        state = obs[:, :args.obs_dim]
        commanded_goal = obs[:, args.obs_dim:]
        commanded_g_repr = sf_encoder.apply(training_state.critic_state.params["sf_encoder"], commanded_goal)

        means, log_stds = actor.apply(training_state.actor_state.params, state, commanded_g_repr)
        # means, log_stds = actor.apply(actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return training_state, nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )


    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            training_state, env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            training_state, env_state, transition = actor_step(training_state, env, env_state, current_key,
                                                               extra_fields=("seed",))
            return (training_state, env_state, next_key), transition

        (training_state, env_state, _), data = jax.lax.scan(f, (training_state, env_state, key), (),
                                                            length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return training_state, env_state, buffer_state


    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            training_state, env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                key,

            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[
            0]


    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            sag_encoder_params, sf_encoder_params = critic_params["sag_encoder"], critic_params["sf_encoder"]

            obs = transitions.observation  # expected_shape = batch_size, obs_size + goal_size
            state = obs[:, :args.obs_dim]
            future_goal = transitions.extras["future_goal"]
            # observation = jnp.concatenate([state, future_goal], axis=1)

            sf_repr = sf_encoder.apply(sf_encoder_params, future_goal)
            means, log_stds = actor.apply(actor_params, state, sf_repr)
            # means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)  # dimension = B

            sag_repr = sag_encoder.apply(sag_encoder_params, state, action, future_goal)
            g_repr = sf_encoder.apply(sf_encoder_params, future_goal)

            qf_pi = -jnp.sqrt(jnp.sum((sag_repr - g_repr) ** 2, axis=-1))

            actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - (qf_pi))

            return actor_loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)

        (actor_l, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params, training_state.critic_state.params,
            training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alpha_l, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actor_l,
            "alpha_loss": alpha_l,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics


    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, transitions, key):
            sag_encoder_params, sf_encoder_params = critic_params["sag_encoder"], critic_params["sf_encoder"]

            state = transitions.observation[:, :args.obs_dim]
            action = transitions.action
            commanded_goal = transitions.extras["commanded_goal"]
            future_goal = transitions.extras["future_goal"]

            sag_repr = sag_encoder.apply(sag_encoder_params, state, action, commanded_goal)
            sf_repr = sf_encoder.apply(sf_encoder_params, future_goal)

            # InfoNCE
            logits = -jnp.sqrt(jnp.sum((sag_repr[:, None, :] - sf_repr[None, :, :]) ** 2, axis=-1))  # shape = BxB
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss, (logsumexp, correct, logits_pos, logits_neg)

        (critic_l, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": critic_l,
        }

        return training_state, metrics


    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)

        return (training_state, key,), metrics


    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # update buffer
        training_state, env_state, buffer_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state,), metrics


    @jax.jit
    def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
    ):
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (
        training_state, env_state, buffer_state, key), (), length=args.num_training_steps_per_epoch)

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics


    '''Setting up evaluator'''
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    print('prefilling replay buffer....')
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    training_walltime = 0
    print('starting training....')
    for ne in range(args.num_epochs):

        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state,
                                                                          epoch_key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics, stats = evaluator.run_evaluation(training_state, metrics)
        if args.track:
            # plot trajectories
            import matplotlib.pyplot as plt

            fig = plot_trajectories(args.num_eval_vis, stats, use_planner=False)
            wandb.log({"evaluation_trajectory": wandb.Image(fig)}, step=ne)
            plt.close(fig)

        print(metrics)

        if args.checkpoint and ne % args.num_epochs_per_checkpoint == 0:
            # Save current policy and critic params.
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )
            path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
            save_params(path, params)

        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == 'offline':
                trigger_sync()

    if args.checkpoint:
        # Save current policy and critic params.
        params = (
            training_state.alpha_state.params,
            training_state.actor_state.params,
            training_state.critic_state.params,
        )
        path = f"{save_path}/final.pkl"
        save_params(path, params)

        if args.checkpoint_final_rb:
            path = f"{save_path}/final_rb.pkl"
            save_params(path, buffer_state)

    render(training_state.actor_state, env, save_path, args.exp_name,
           wandb_track=args.track)

# (50000000 - 1024 x 1000) / 50 x 1024 x 62 = 15        #number of actor steps per epoch (which is equal to the number of training steps)
# 1024 x 999 / 256 = 4000                               #number of gradient steps per actor step 
# 1024 x 62 / 4000 = 16                                 #ratio of env steps per gradient step
