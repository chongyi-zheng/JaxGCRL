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
from evaluator import CrlPlanningEvaluator
from utils import plot_trajectories


@dataclass
class Args:
    debug: bool = False
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
    """number of candidate waypoints"""
    num_candidates: int = 1000
    """whether to use planner during training"""
    train_planner: bool = False
    """whether to use planner during evaluation"""
    eval_planner: bool = False

    # to be filled in runtime
    env_steps_per_actor_step: int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps: int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps: int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch: int = 0
    """the number of training steps per epoch(computed in runtime)"""


class SA_encoder(nn.Module):
    repr_dim: int = 64
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = jnp.concatenate([s, a], axis=-1)
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


class S_encoder(nn.Module):
    repr_dim: int = 64
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(g)
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
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

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

@flax.struct.dataclass
class PlanningState:
    candidate_obs: jnp.ndarray


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


def main(args):
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
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, s_key, g_key = jax.random.split(key, 8)

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
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder(repr_dim=args.repr_dim)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    s_encoder = S_encoder(repr_dim=args.repr_dim)
    s_encoder_params = s_encoder.init(s_key, np.ones([1, args.obs_dim]))
    g_encoder = S_encoder(repr_dim=args.repr_dim)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    # c = jnp.asarray(0.0, dtype=jnp.float32)
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params,
                "s_encoder": s_encoder_params,
                "g_encoder": g_encoder_params},
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

    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    planning_state = PlanningState(
        candidate_obs=jnp.zeros((args.num_candidates, args.obs_dim), dtype=jnp.float32),
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

    def energy_fn(x, y):
        energy = -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)

        return energy

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
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
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )


    def planner_step(env_state, planning_state, key):
        # TODO (chong: there seems to have a bug here
        # posterior sampling
        goals = env_state.obs[:, args.obs_dim:]
        states = env_state.obs[:, :args.obs_dim]
        w_candidates = planning_state.candidate_obs  # (N, obs_dim)

        s_encoder_params, g_encoder_params = (
            training_state.critic_state.params["s_encoder"],
            training_state.critic_state.params["g_encoder"]
        )

        s_repr = s_encoder.apply(s_encoder_params, states)
        w_repr = g_encoder.apply(g_encoder_params, w_candidates[:, args.goal_start_idx:args.goal_end_idx])
        sw_logits = energy_fn(s_repr[:, None], w_repr[None, :])  # (B, N)

        w_repr = s_encoder.apply(s_encoder_params, w_candidates)
        g_repr = g_encoder.apply(g_encoder_params, goals)
        wg_logits = energy_fn(w_repr[:, None], g_repr[None, :])  # (N, B)

        # resampling importance sampling
        log_scores = sw_logits + wg_logits.transpose(1, 0)  # (B, N)
        # scores = jax.nn.softmax(scores, axis=-1)
        # w_idxs = jnp.argmax(scores, axis=-1)  # (B,)
        w_idxs = jax.random.categorical(key, log_scores, axis=-1)
        waypoint = w_candidates[w_idxs][:, args.goal_start_idx:args.goal_end_idx]  # (B,)

        env_state_with_waypoint = env_state.replace(
            obs=jnp.concatenate([env_state.obs[:, :args.obs_dim], waypoint], axis=-1))
        return env_state_with_waypoint

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, planning_state, key, use_planner=False):
        @jax.jit
        def f(carry, unused_t):
            env_state, planning_state, current_key = carry
            next_key, planning_key = jax.random.split(current_key)

            env_state = jax.lax.cond(
                use_planner,
                planner_step,
                lambda es, ps, k: es,
                env_state, planning_state, planning_key
            )
            env_state, transition = actor_step(training_state, env, env_state, current_key,
                                               extra_fields=("truncation", "seed"))
            return (env_state, planning_state, next_key), transition

        # @jax.jit
        # def sample_transitions(replay_buffer, buffer_state, key):
        #     buffer_state, transitions = replay_buffer.sample(buffer_state)
        #
        #     # process transitions for training
        #     key, batch_key, perm_key = jax.random.split(key, 3)
        #     batch_keys = jax.random.split(batch_key, transitions.observation.shape[0])
        #     transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
        #         (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        #     )
        #
        #     transitions = jax.tree_util.tree_map(
        #         lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
        #         transitions,
        #     )
        #     permutation = jax.random.permutation(perm_key, len(transitions.observation))
        #     transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        #     transitions = jax.tree_util.tree_map(
        #         lambda x: x[:args.num_candidates],
        #         transitions
        #     )
        #
        #     return buffer_state, transitions

        # key, sampling_key = jax.random.split(key, 2)
        # buffer_state, waypoint_transitions = sample_transitions(
        #     replay_buffer, buffer_state, sampling_key)
        #
        # planning_state = planning_state.replace(
        #     candidate_obs=waypoint_transitions.observation[:, :args.obs_dim],
        # )

        (env_state, planning_state, _), data = jax.lax.scan(
            f, (env_state, planning_state, key), (),
            length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state, planning_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, planning_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, planning_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state, planning_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                planning_state,
                key,
                use_planner=False,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, planning_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, planning_state, key), (),
                            length=args.num_prefill_actor_steps)[0]


    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation  # expected_shape = batch_size, obs_size + goal_size
            state = obs[:, :args.obs_dim]
            goal = transitions.extras["future_goal"]
            observation = jnp.concatenate([state, goal], axis=1)

            means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)  # dimension = B

            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)

            qf_pi = energy_fn(sa_repr, g_repr)

            actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - (qf_pi))

            return actor_loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)

        (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params, training_state.critic_state.params,
            training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics


    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, s_encoder_params, g_encoder_params = (
                critic_params["sa_encoder"],
                critic_params["s_encoder"],
                critic_params["g_encoder"]
            )

            obs = transitions.observation[:, :args.obs_dim]
            future_goal = transitions.extras["future_goal"]
            future_state = transitions.extras["future_state"]
            action = transitions.action

            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, future_goal)
            s_repr = s_encoder.apply(s_encoder_params, obs)
            sf_repr = s_encoder.apply(s_encoder_params, future_state)

            # InfoNCE
            I = jnp.eye(obs.shape[0])
            sag_logits = energy_fn(sa_repr[:, None, :], g_repr[None, :, :])
            ssf_logits = energy_fn(s_repr[:, None, :], sf_repr[None, :, :])
            sag_critic_loss = jnp.mean(optax.softmax_cross_entropy(sag_logits, I))
            ssf_critic_loss = jnp.mean(optax.softmax_cross_entropy(ssf_logits, I))
            critic_loss = sag_critic_loss + ssf_critic_loss

            # logsumexp regularisation
            sag_logsumexp = jax.nn.logsumexp(sag_logits + 1e-6, axis=1)
            # sg_logsumexp = jax.nn.logsumexp(sg_logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_penalty_coeff * jnp.mean(sag_logsumexp ** 2)
            # critic_loss += args.logsumexp_penalty_coeff * jnp.mean(sg_logsumexp ** 2)

            sag_correct = jnp.argmax(sag_logits, axis=1) == jnp.argmax(I, axis=1)
            sag_logits_pos = jnp.sum(sag_logits * I) / jnp.sum(I)
            sag_logits_neg = jnp.sum(sag_logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss, (critic_loss, sag_critic_loss, ssf_critic_loss,
                                 sag_logsumexp.mean(), sag_correct, sag_logits_pos, sag_logits_neg)

        (loss, (critic_loss, sag_critic_loss, ssf_critic_loss,
            sag_logsumexp, sag_correct, sag_logits_pos, sag_logits_neg)), grad = jax.value_and_grad(
            critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)

        metrics = {
            "critic_loss": critic_loss,
            "sag_critic_loss": sag_critic_loss,
            "ssf_critic_loss": ssf_critic_loss,
            "sag_logsumexp": sag_logsumexp,
            "sag_correct": sag_correct,
            "sag_logits_pos": sag_logits_pos,
            "sag_logits_neg": sag_logits_neg,
        }

        return training_state, metrics

    @jax.jit
    def update_planning_state(transitions, training_state, planning_state):
        planning_state = planning_state.replace(
            candidate_obs=transitions.observation[:, :args.obs_dim],
        )

        return training_state, planning_state

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key, = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)

        return (training_state, key,), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, planning_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # update buffer
        env_state, buffer_state, planning_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            planning_state,
            experience_key1,
            use_planner=args.train_planner,
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
        waypoint_transitions = jax.tree_util.tree_map(
            lambda x: x[:args.num_candidates],
            transitions,
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        # update planning state
        training_state, planning_state = update_planning_state(
            waypoint_transitions, training_state, planning_state)

        return (training_state, env_state, buffer_state, planning_state), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        planning_state,
        key,
    ):
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, ps, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs, ps), metrics = training_step(ts, es, bs, ps, train_key)
            return (ts, es, bs, ps, k), metrics

        (training_state, env_state, buffer_state, planning_state, key), metrics = jax.lax.scan(f, (
            training_state, env_state, buffer_state, planning_state, key), (), length=args.num_training_steps_per_epoch)

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, planning_state, metrics


    key, prefill_key = jax.random.split(key, 2)

    training_state, env_state, buffer_state, planning_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, planning_state, prefill_key
    )

    '''Setting up evaluator'''
    evaluator = CrlPlanningEvaluator(
        deterministic_actor_step,
        planner_step if args.eval_planner else lambda s, ps, k: s,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    training_walltime = 0
    print('starting training....')
    for ne in range(args.num_epochs):

        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, planning_state, metrics = training_epoch(
            training_state, env_state, buffer_state, planning_state, epoch_key)

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

        metrics, stats = evaluator.run_evaluation(training_state, planning_state, metrics)
        if args.track:
            # plot trajectories
            import matplotlib.pyplot as plt
            fig = plot_trajectories(args.num_eval_vis, stats, use_planner=args.eval_planner)
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
            training_state.critic_state.params
        )
        path = f"{save_path}/final.pkl"
        save_params(path, params)

        if args.checkpoint_final_rb:
            path = f"{save_path}/final_rb.pkl"
            save_params(path, buffer_state)

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

    render(training_state.actor_state, env, save_path, args.exp_name,
           wandb_track=args.track)

if __name__ == "__main__":
    args = tyro.cli(Args)
    with jax.disable_jit(args.debug):
        main(args)

# (50000000 - 1024 x 1000) / 50 x 1024 x 62 = 15        #number of actor steps per epoch (which is equal to the number of training steps)
# 1024 x 999 / 256 = 4000                               #number of gradient steps per actor step 
# 1024 x 62 / 4000 = 16                                 #ratio of env steps per gradient step
