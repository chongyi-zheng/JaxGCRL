import jax
import jax.numpy as jnp


def bellman_ford(adj_matrix, src):
    """
    Vectorized Bellman–Ford algorithm finding single source shortest paths (SSSP).
    """
    n = adj_matrix.shape[0]
    distances = jnp.full(n, jnp.inf)
    distances = distances.at[src].set(0)

    # Initialize predecessors for path reconstruction
    predecessors = -jnp.ones(n, dtype=jnp.int32)

    # Edges exist where the adjacency matrix is non-zero
    edges_mask = adj_matrix != 0

    def cond_f(state):
        idx, distances, predecessors, updates_made = state
        # Continue if iterations are left and updates were made
        return (idx < n - 1) & updates_made

    def f(state):
        idx, distances, predecessors, _ = state

        valid_nodes = distances != jnp.inf
        valid_edges = edges_mask & valid_nodes[:, jnp.newaxis]

        # Calculate potential distances only for valid edges
        potential_distances = distances[:, jnp.newaxis] + adj_matrix
        potential_distances = jnp.where(valid_edges, potential_distances, jnp.inf)

        # Find the minimum potential distances and their predecessors
        min_potential_distances = jnp.min(potential_distances, axis=0)
        potential_predecessors = jnp.argmin(potential_distances, axis=0)

        # Identify where distances can be updated
        updates_mask = min_potential_distances < distances

        # Update distances and predecessors where applicable
        distances = jnp.where(updates_mask, min_potential_distances, distances)
        predecessors = jnp.where(updates_mask, potential_predecessors, predecessors)

        # Check if any updates were made
        updates_made = jnp.any(updates_mask)

        return (idx + 1, distances, predecessors, updates_made)

    # Initialize the loop state
    idx = 0
    updates_made = True
    state = (idx, distances, predecessors, updates_made)

    # Use lax.while_loop for proper JIT compilation
    state = jax.lax.while_loop(cond_f, f, state)
    _, distances, predecessors, _ = state

    # Check for negative-weight cycles
    valid_nodes = distances != jnp.inf
    valid_edges = edges_mask & valid_nodes[:, jnp.newaxis]
    potential_distances = distances[:, jnp.newaxis] + adj_matrix
    potential_distances = jnp.where(valid_edges, potential_distances, jnp.inf)
    min_potential_distances = jnp.min(potential_distances, axis=0)

    has_negative_cycle = jnp.any(min_potential_distances < distances)

    return distances, predecessors, has_negative_cycle


def get_shortest_path(dists, predecessors, start, end):
    max_path_length = predecessors.shape[0]  # Maximum possible path length

    # Initialize the path array with -1
    path = -jnp.ones(max_path_length, dtype=jnp.int32)
    dists_to_end = jnp.ones(max_path_length, dtype=jnp.float32) * jnp.inf

    # Start from the end node
    current = end
    index = max_path_length - 1

    # Define the loop condition and body functions
    def cond_f(state):
        current, index, _, _ = state
        return (current != -1) & (index >= 0)

    def f(state):
        current, index, path, dists_to_end = state
        path = path.at[index].set(current)
        dist_to_end = dists[end] - dists[current]
        dists_to_end = dists_to_end.at[index].set(dist_to_end)
        current = predecessors[current]
        index -= 1
        return current, index, path, dists_to_end

    # Run the loop using lax.while_loop
    state = (current, index, path, dists_to_end)
    current, index, path, dists_to_end = jax.lax.while_loop(
        cond_f, f, state)

    # Compute the valid start index
    valid_start_index = index + 1

    # If the path does not start with 'start', then no path exists
    # Replace path with an array of -1s of the same length
    path, dists_to_end = jax.lax.cond(
        path[valid_start_index] == start,
        lambda p, d: (p, d),
        lambda p, d: (-jnp.ones_like(p, dtype=jnp.int32), jnp.ones_like(p, dtype=jnp.float32) * jnp.inf),
        path, dists_to_end
    )

    return path, dists_to_end, valid_start_index


def bellman_ford_with_midwaypoints(adj_matrix, src):
    n = adj_matrix.shape[0]
    distances = jnp.full(n, jnp.inf)
    distances = distances.at[src].set(0)

    # Initialize predecessors for path reconstruction
    predecessors = -jnp.ones(n, dtype=jnp.int32)

    # Edges exist where the adjacency matrix is non-zero
    edges_mask = adj_matrix != 0

    def cond_fun(state):
      idx, distances, predecessors, updates_made = state
      # Continue if iterations are left and updates were made
      return (idx < n - 1) & updates_made

    def body_fun(state):
      idx, distances, predecessors, _ = state

      valid_nodes = distances != jnp.inf
      valid_edges = edges_mask & valid_nodes[:, jnp.newaxis]

      # Calculate potential distances only for valid edges
      potential_distances = distances[:, jnp.newaxis] + adj_matrix
      potential_distances = jnp.where(valid_edges, potential_distances, jnp.inf)

      # Find the minimum potential distances and their predecessors
      min_potential_distances = jnp.min(potential_distances, axis=0)
      potential_predecessors = jnp.argmin(potential_distances, axis=0)

      # Identify where distances can be updated
      updates_mask = min_potential_distances < distances

      # Update distances and predecessors where applicable
      distances = jnp.where(updates_mask, min_potential_distances, distances)
      predecessors = jnp.where(updates_mask, potential_predecessors, predecessors)

      # Check if any updates were made
      updates_made = jnp.any(updates_mask)

      return (idx + 1, distances, predecessors, updates_made)

    # Initialize the loop state
    idx = 0
    updates_made = True
    state = (idx, distances, predecessors, updates_made)

    # Use lax.while_loop for proper JIT compilation
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    _, distances, predecessors, _ = state

    # Check for negative-weight cycles
    valid_nodes = distances != jnp.inf
    valid_edges = edges_mask & valid_nodes[:, jnp.newaxis]
    potential_distances = distances[:, jnp.newaxis] + adj_matrix
    potential_distances = jnp.where(valid_edges, potential_distances, jnp.inf)
    min_potential_distances = jnp.min(potential_distances, axis=0)

    has_negative_cycle = jnp.any(min_potential_distances < distances)

    path, _, start_index = jax.vmap(get_shortest_path, in_axes=(None, None, None, 0))(
      distances, predecessors, src, jnp.arange(n))

    mid_index = (start_index + n) // 2
    mid_waypoints = path[jnp.arange(n), mid_index]

    return distances, predecessors, mid_waypoints, has_negative_cycle
