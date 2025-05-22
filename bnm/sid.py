import numpy as np
from bnm.utils import graph_to_matrix, get_undirected_components_with_isolates
from networkx.algorithms.chordal import is_chordal

##### This code is a translation of the SID R package by Jonas Peters.
##### All credit for the original algorithm and implementation goes to the original author.

def compute_path_matrix(adj_mat):
    p = adj_mat.shape[1]
    path_matrix = np.eye(p) + adj_mat
    
    k = int(np.ceil(np.log(p) / np.log(2)))
    
    for _ in range(k):
        path_matrix = np.dot(path_matrix, path_matrix)
    path_matrix = (path_matrix > 0).astype(int)
    return path_matrix

def compute_path_matrix2(adj_mat, cond_set, path_matrix1=None):
    """
    Remove all outgoing edges from cond_set and compute ancestor matrix.
    If cond_set is empty, return path_matrix1 directly.

    Parameters:
        adj_mat (np.ndarray): Adjacency matrix (p x p).
        cond_set (list[int]): Conditioning set (node indices).
        path_matrix1 (np.ndarray or None): Precomputed path matrix.

    Returns:
        np.ndarray: Modified path matrix (p x p) with paths after removing condSet -> *
    """
    p = adj_mat.shape[1]

    if len(cond_set) == 0:
        return path_matrix1

    G_mod = adj_mat.copy()
    G_mod[cond_set, :] = 0  # Remove outgoing edges from nodes in cond_set

    path_matrix2 = np.eye(p) + G_mod
    k = int(np.ceil(np.log(p) / np.log(2)))

    for _ in range(k):
        path_matrix2 = np.dot(path_matrix2, path_matrix2)

    path_matrix2 = (path_matrix2 > 0).astype(int)
    return path_matrix2

def all_dags_intern(adj_full, sub_adj, node_indices, tmp=None):
    """
    Recursively extend an undirected subgraph to all valid DAGs without creating v-structures.

    Parameters:
    - adj_full: full adjacency matrix (NumPy array)
    - sub_adj: symmetric submatrix representing the undirected component
    - node_indices: list/array of indices corresponding to sub_adj
    - tmp: accumulator list for resulting DAGs

    Returns:
    - tmp: list of DAGs as NumPy arrays
    """
    if np.any((sub_adj + sub_adj.T) == 1):
        raise ValueError("Submatrix is not fully undirected (not symmetric).")

    if np.sum(sub_adj) == 0:
        # All edges have been oriented
        if tmp is None:
            tmp = [adj_full.copy().flatten()]
        elif not any(np.array_equal(adj_full, t) for t in tmp):
            tmp.append(adj_full.copy().flatten())
        return tmp

    # Candidate sink nodes: must have at least one neighbor
    candidate_sinks = np.where(sub_adj.sum(axis=0) > 0)[0]

    for sink_local_idx in candidate_sinks:
        neighbors_local_idx = np.where(sub_adj[sink_local_idx] == 1)[0]

        # Check if neighbors form a clique
        if neighbors_local_idx.size > 0:
            subgraph_neighbors = sub_adj[np.ix_(neighbors_local_idx, neighbors_local_idx)]
            if not np.all(subgraph_neighbors + np.eye(len(neighbors_local_idx))):
                continue

        # Orient edges toward sink
        adj_new = adj_full.copy()
        sink_global_idx = node_indices[sink_local_idx]
        for neighbor_local_idx in neighbors_local_idx:
            neighbor_global_idx = node_indices[neighbor_local_idx]
            adj_new[neighbor_global_idx, sink_global_idx] = 1
            adj_new[sink_global_idx, neighbor_global_idx] = 0

        # Remove sink from subgraph
        mask = np.ones(len(sub_adj), dtype=bool)
        mask[sink_local_idx] = False
        new_sub_adj = sub_adj[np.ix_(mask, mask)]
        new_node_indices = node_indices[mask]

        tmp = all_dags_intern(adj_new, new_sub_adj, new_node_indices, tmp)

    return np.array([dag.flatten() for dag in tmp])

def all_dags_jonas(adj_matrix, node_indices):
    """
    Wrapper to extend the undirected component of a DAG into all consistent DAGs.

    Parameters:
    - adj_matrix: full adjacency matrix (NumPy array)
    - node_indices: list/array of indices for the undirected component

    Returns:
    - -1 if the component is not fully undirected (i.e. asymmetric edges exist)
    - otherwise: list of adjacency matrices (NumPy arrays) representing all valid DAGs
    """
    sub_adj = adj_matrix[np.ix_(node_indices, node_indices)]

    if np.any((sub_adj + sub_adj.T) == 1):
        # The subgraph has at least one directed edge (i.e., not symmetric)
        return -1

    return all_dags_intern(adj_matrix.copy(), sub_adj, np.array(node_indices), tmp=None)


def dsepadj(a_mat, i, cond_set):
    """
    Compute the reachability of nodes from a source node `i` in a DAG
    given a conditioning set `cond_set`. This function determines if nodes are reachable 
    through directed or non-directed paths based on d-separation criteria.

    Parameters
    ----------
    adj_mat : np.ndarray
        Adjacency matrix of the DAG.
    i : int
        The index of the source node for which reachability is calculated.
    cond_set : list of int
        A list of node indices representing the conditioning set, 
        which blocks certain paths during traversal.

    Returns
    -------
    result : dict
        A dictionary with two keys:
        
        - 'reachable_nodes': np.ndarray
            A boolean array of size `p`, where each element indicates whether the node is 
            reachable from the source node `i` under d-separation rules.
        
        - 'reachable_on_non_causal_path': np.ndarray
            A boolean array of size `p`, where each element indicates whether the node is 
            reachable from the source node `i` via non-directed paths 
            that are not blocked by the conditioning set.
    """
    adj_mat = a_mat.copy()
    path_matrix = compute_path_matrix(adj_mat)
    path_matrix2 = compute_path_matrix2(adj_mat, cond_set, path_matrix)
    if len(cond_set) == 0:
        anc_of_cond_set = np.array([], dtype=int)
    elif len(cond_set) == 1:
        anc_of_cond_set = np.where(path_matrix[:, cond_set[0]] > 0)[0]
    else:
        anc_of_cond_set = np.where(np.sum(path_matrix[:, cond_set], axis=1) > 0)[0]
    anc_of_cond_set

    p = adj_mat.shape[1]
    reachability_matrix = np.zeros((2 * p, 2 * p), dtype=int)
    reachable_on_non_causal_path_later = np.zeros((2, 2), dtype=int)

    reachable_nodes = np.zeros(2 * p, dtype=int)
    reachable_on_non_causal_path = np.zeros(2 * p, dtype=int)
    already_checked = np.zeros(p, dtype=int)
    k = 2
    to_check = []

    # Direct children
    reachable_ch = np.where(adj_mat[i, :] == 1)[0]
    if len(reachable_ch) > 0:
        to_check.extend(reachable_ch.tolist())
        reachable_nodes[reachable_ch] = 1 
        adj_mat[i, reachable_ch] = 0 

    # Direct parents
    reachable_pa = np.where(adj_mat[:, i] == 1)[0]
    if len(reachable_pa) > 0:
        to_check.extend(reachable_pa.tolist())
        reachable_nodes[reachable_pa + p] = 1
        reachable_on_non_causal_path[reachable_pa + p] = 1
        adj_mat[reachable_pa, i] = 0

    k = -1

    while k < (len(to_check)-1):
        k += 1
        a1 = to_check[k]

        if not already_checked[a1]:
            current_node = a1
            already_checked[a1] = a1

            parents = np.where(adj_mat[:, current_node] == 1)[0]
            pa1 = np.setdiff1d(parents, cond_set)
            reachability_matrix[pa1, current_node] = 1
            reachability_matrix[pa1 + p, current_node] = 1

            if current_node in anc_of_cond_set:
                reachability_matrix[current_node, parents + p] = 1
                if path_matrix2[i, current_node] > 0:
                    new_rows = np.column_stack((np.full(len(parents), current_node), parents))
                    reachable_on_non_causal_path_later = np.vstack((reachable_on_non_causal_path_later, new_rows))
                new_to_check = np.setdiff1d(parents, np.where(already_checked)[0])
                to_check.extend(new_to_check.tolist())

            if current_node not in cond_set:
                reachability_matrix[current_node + p, parents + p] = 1
                new_to_check = np.setdiff1d(parents, np.where(already_checked)[0])
                to_check.extend(new_to_check.tolist())
            children = np.where(adj_mat[current_node, :] == 1)[0]
            ch1 = np.setdiff1d(children, cond_set)
            reachability_matrix[ch1 + p, current_node + p] = 1

            ch2 = np.intersect1d(children, anc_of_cond_set)
            reachability_matrix[ch2, current_node + p] = 1
            ch2b = np.intersect1d(ch2, np.where(path_matrix2[i, :] > 0)[0])
            if len(ch2b) > 0:
                new_rows = np.column_stack((ch2b, np.full(len(ch2b), current_node)))
                reachable_on_non_causal_path_later = np.vstack((reachable_on_non_causal_path_later, new_rows))
            if current_node not in cond_set:
                reachability_matrix[current_node, children] = 1
                reachability_matrix[current_node + p, children] = 1
                new_to_check = np.setdiff1d(children, np.where(already_checked)[0])
                to_check.extend(new_to_check.tolist())

    reachability_matrix = compute_path_matrix(reachability_matrix)
    ttt2 = np.where(reachable_nodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachability_matrix[ttt2[0], :] > 0)[0]
    else:
        tt2 = np.where(np.sum(reachability_matrix[ttt2, :], axis=0) > 0)[0]

    reachable_nodes[tt2] = 1
    ttt = np.where(reachable_on_non_causal_path == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
    else:
        tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]

    reachable_on_non_causal_path[tt] = 1

    if reachable_on_non_causal_path_later.shape[0] > 2:
        for kk in range(2, reachable_on_non_causal_path_later.shape[0]):
            reachable_through = reachable_on_non_causal_path_later[kk, 0]
            new_reachable = reachable_on_non_causal_path_later[kk, 1]

            reachable_on_non_causal_path[new_reachable + p] = 1

            reachability_matrix[new_reachable, reachable_through] = 0
            reachability_matrix[new_reachable, reachable_through + p] = 0
            reachability_matrix[new_reachable + p, reachable_through] = 0
            reachability_matrix[new_reachable + p, reachable_through + p] = 0

        ttt = np.where(reachable_on_non_causal_path == 1)[0]

        if len(ttt) == 1:
            tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
        else:
            tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]

        reachable_on_non_causal_path[tt] = 1

    result = {
        'reachable_nodes': np.sum(np.column_stack((reachable_nodes[:p], reachable_nodes[p:])), axis=1) > 0,
        'reachable_on_non_causal_path': np.sum(np.column_stack((reachable_on_non_causal_path[:p], reachable_on_non_causal_path[p:])), axis=1) > 0
    }
    return result


def sid_metric(true_dag, est_dag):
    """
    SID quantifes the closeness between two DAGs in terms of their corresponding causal
    inference statements. It is well-suited for evaluating graphs that are used for computing
    interventions.
    https://doi.org/10.48550/arXiv.1306.1043
    This implementation is a translation of the R package SID originally developed by Jonas Peters. 
    All credit for the original methodology and implementation goes to the author.

    Parameters
    ----------
    true_dag : networkx.DiGraph
        The ground-truth DAG representing the actual causal relationships.
    
    est_dag : networkx.DiGraph
        The estimated DAG or CPDAG that is being compared to the true DAG.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        
        - 'sid': int
            The Structural Intervention Distance (SID) value representing the total number of incorrect
            interventions in the estimated DAG compared to the true DAG.
        
        - 'sidUpperBound': int
            The upper bound for the SID, indicating the maximum possible intervention errors.
        
        - 'sidLowerBound': int
            The lower bound for the SID, indicating the minimum possible intervention errors.
        
        - 'incorrectMat': np.ndarray
            A matrix of shape (p, p), where `p` is the number of nodes in the DAG, containing binary values 
            indicating where incorrect interventions were identified. `1` indicates an incorrect intervention, 
            `0` indicates correctness.
    """


    g1, _ = graph_to_matrix(true_dag)
    g2, _ = graph_to_matrix(est_dag)
    p = g1.shape[0]
    incorrect_int = np.zeros(shape=(g1.shape))
    correct_int = np.zeros(shape=(g1.shape))
    min_total = 0
    max_total = 0
    num_checks = 0
    path_matrix = compute_path_matrix(g1)
    conn_comp = get_undirected_components_with_isolates(est_dag)
    gp_is_essential_graph = True # estimated DAG is CPDAG
    node_names = list(true_dag.nodes())
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    for comp in conn_comp:
        comp_nodes = list(comp)
        if len(comp_nodes) > 1:
            subgraph = est_dag.subgraph(comp_nodes).to_undirected()
            if not is_chordal(subgraph):
                gp_is_essential_graph = False
    for comp in conn_comp:
        node_indices = [name_to_idx[name] for name in list(comp)]
        if len(node_indices) > 0:
            if gp_is_essential_graph:
                if len(node_indices) > 1:
                    mmm = all_dags_jonas(g2, node_indices)
                else:
                    # Only one node in component — trivially already a DAG
                    mmm = np.array([g2.copy().flatten()])

                if isinstance(mmm, int) and mmm == -1:
                    gp_is_essential_graph = False
                    mmm = np.array([g2.copy().flatten()])

                if len(mmm) == 0:
                    gp_is_essential_graph = False
                else:
                    # Initialize counter for incorrect interventions (to be used later)
                    incorrect_sum = np.zeros(len(mmm))
        for i in node_indices:
            # Identify parents of `i` in the true graph
            pa_g = np.where(g1[:, i] == 1)[0]

            # Certain parents in the estimated graph
            certain_pa_gp = np.where((g2[:, i] * (1 - g2[i, :])) == 1)[0]
            # Possible parents (bidirectional edges in CPDAG)
            possible_pa_gp = np.where((g2[:, i] * g2[i, :]) == 1)[0]
            if not gp_is_essential_graph: # this to work with
                # All combinations of parents
                maxcount = 2 ** len(possible_pa_gp)
                unique_rows = np.arange(maxcount)
                # Build matrix of parent combinations
                base_flat = g2.T.flatten()
                mmm = np.tile(base_flat, (maxcount, 1))

                # Create binary combinations (like expand.grid in R)
                grid = np.array(np.meshgrid(*[[0, 1]] * len(possible_pa_gp))).T.reshape(-1, len(possible_pa_gp))
                
                # Update parent sets in `mmm`
                for row_idx, combo in enumerate(grid):
                    for j, val in enumerate(combo):
                        mmm[row_idx, i + possible_pa_gp[j] * p] = val
                
                # Initialize incorrect sum counter
                incorrect_sum = np.zeros(maxcount)

            else:
                if len(mmm) > 1:
                    # If there are multiple DAGs, extract unique parent sets for node `i`
                    all_parents_of_i = [i + k * p for k in range(p)]
                    parent_sets = mmm[:, all_parents_of_i]
                    # Find unique configurations
                    _, unique_indices = np.unique(parent_sets, axis=0, return_index=True)
                    unique_rows = np.sort(unique_indices)
                    maxcount = len(unique_rows)
                else:
                    maxcount = 1
                    unique_rows = np.array([0])
            count = 0  # Start the counter

            # Iterate through all expansions. In case of a CPDAG, only go through uniqueRows.
            while count < maxcount:
                if maxcount == 1:
                    # Only one parent set; it is the certain parents in the essential graph
                    pa_gp = certain_pa_gp
                else:                    
                    # Extract the unique parent set from `mmm` and reshape it to a matrix
                    gp_new_flat = mmm[unique_rows[count], :]
                    gp_new = gp_new_flat.reshape((p, p))  # Transpose to match R's `t(matrix(...))`
                    
                    # Find parents of `i` in the new configuration
                    pa_gp = np.where(gp_new[:, i] == 1)[0]
                check_all_d_sep = dsepadj(g1, i, pa_gp)

                # Increment the counter for checks
                num_checks += 1

                # Extract reachable paths that are not on a causal path
                reachable_w_out_causal_path = check_all_d_sep["reachable_on_non_causal_path"]

                for j in range(p):
                    if i != j:
                        finished = False
                        ij_g_null = False
                        ij_gp_null = False

                        if path_matrix[i, j] == 0:
                            ij_g_null = True

                        if j in pa_gp:
                            ij_gp_null = True

                        if ij_g_null and ij_gp_null:
                            finished = True
                            correct_int[i, j] = 1

                        if ij_gp_null and not ij_g_null:
                            incorrect_int[i, j] = 1
                            incorrect_sum[unique_rows[count]] += 1

                            # Add one to all incorrect_sum entries with the same set of parents of i
                            all_others = [x for x in range(mmm.shape[0]) if x not in unique_rows]
                            if len(all_others) > 1:
                                ind_in_all_others = [idx for idx in all_others if sum(~(mmm[unique_rows[count], all_parents_of_i] ^ mmm[idx, all_parents_of_i])) == p]
                                if ind_in_all_others:
                                    incorrect_sum[ind_in_all_others] += 1
                            if len(all_others) == 1:
                                if sum(~(mmm[unique_rows[count], all_parents_of_i] ^ mmm[all_others[0], all_parents_of_i])) == p:
                                    incorrect_sum[all_others[0]] += 1
                            finished = True

                        if not finished and set(pa_gp) == set(pa_g):
                            finished = True
                            correct_int[i, j] = 1


                        if not finished:
                            if path_matrix[i, j] > 0:
                                chi_caus_path = np.where((g1[i, :] & path_matrix[:, j]))[0]
                                if len(chi_caus_path) > 0 and len(pa_gp) > 0:
                                    submatrix = path_matrix[np.ix_(chi_caus_path, pa_gp)]
                                    if np.sum(submatrix) > 0:
                                        incorrect_int[i, j] = 1
                                        incorrect_sum[unique_rows[count]] += 1
                                        all_others = [x for x in range(mmm.shape[0]) if x not in unique_rows]
                                        if len(all_others) > 1:
                                            ind_in_all_others = [idx for idx in all_others if sum(~(mmm[unique_rows[count], all_parents_of_i] ^ mmm[idx, all_parents_of_i])) == p]
                                            if ind_in_all_others:
                                                incorrect_sum[ind_in_all_others] += 1
                                        if len(all_others) == 1:
                                            if sum(~(mmm[unique_rows[count], all_parents_of_i] ^ mmm[all_others[0], all_parents_of_i])) == p:
                                                incorrect_sum[all_others[0]] += 1

                                        finished = True

                            if not finished:
                                if reachable_w_out_causal_path[j] == 1:
                                    incorrect_int[i, j] = 1
                                    incorrect_sum[unique_rows[count]] += 1
                                    all_others = [x for x in range(mmm.shape[0]) if x not in unique_rows]
                                    if len(all_others) > 1:
                                        ind_in_all_others = [idx for idx in all_others if sum(~(mmm[unique_rows[count], all_parents_of_i] ^ mmm[idx, all_parents_of_i])) == p]
                                        if ind_in_all_others:
                                            incorrect_sum[ind_in_all_others] += 1
                                    if len(all_others) == 1:
                                        if sum(~(mmm[unique_rows[count], all_parents_of_i] ^ mmm[all_others[0], all_parents_of_i])) == p:
                                            incorrect_sum[all_others[0]] += 1
                                else:
                                    correct_int[i, j] = 1
                count += 1

            # Update bounds if graph is not essential
            if not gp_is_essential_graph:
                min_total += np.min(incorrect_sum)
                max_total += np.max(incorrect_sum)
                incorrect_sum = np.zeros_like(incorrect_sum)
        min_total += np.min(incorrect_sum)
        max_total += np.max(incorrect_sum)
        incorrect_sum = np.zeros_like(incorrect_sum)

    return {"sid": np.sum(incorrect_int), "sid_lower_bound": min_total, "sid_upper_bound": max_total, "incorrect_mat": incorrect_int}


