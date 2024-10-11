import matplotlib.pyplot as plt

def plot_trajectories(num_trajs, stats, use_planner=True, axis_lims=(2, 14)):
    assert num_trajs % 4 == 0

    num_rows = num_trajs // 4
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols)
    fig.set_figheight(num_rows * 4)
    fig.set_figwidth(num_cols * 4)

    position = stats['eval/episode_position']
    waypoint_position = stats['eval/episode_waypoint_position']
    target_position = stats['eval/episode_target_position']
    step = stats['eval/episode_steps']

    for row in range(num_rows):
        for col in range(num_cols):
            episode_idx = row * num_cols + col
            episode_step = step[episode_idx]
            episode_pos = position[episode_idx, :episode_step]
            episode_waypoint_pos = waypoint_position[episode_idx, :episode_step]
            episode_target_pos = target_position[episode_idx, 0]
            ax = axes[row, col]

            ax.scatter(episode_pos[0, 0], episode_pos[0, 1], s=8.0, marker='x', color='red',
                       label='start')
            ax.plot(episode_pos[:, 0], episode_pos[:, 1])
            ax.scatter(episode_target_pos[0], episode_target_pos[1], s=8.0, marker='*', color='green',
                       label='goal')
            goal_circle = plt.Circle(episode_target_pos, 0.9, color='green',
                                     fill=False)  # the radius of the circle in the html video is 0.9
            ax.add_patch(goal_circle)
            if use_planner:
                ax.scatter(episode_waypoint_pos[:, 0], episode_waypoint_pos[:, 1],
                           s=8.0, marker='s', color='orange', label='waypoint')
            ax.set_xlim([axis_lims[0], axis_lims[1]])
            ax.set_ylim([axis_lims[0], axis_lims[1]])
            ax.legend()

    plt.tight_layout()

    return fig
