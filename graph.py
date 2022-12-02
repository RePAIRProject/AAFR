
import numpy as np
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def add_nodes(G, base_node, indices, distance, weights, threshold):

    """
    Adds a set of nodes and weighted edges based on pairs of indices
    between base_node and all entries in indices. Each node pair shares an
    edge with weight equal to the distance between both nodes.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object to which all nodes/edges will be added.
    base_node : int
        Base node's id to be added. All other nodes will be paired with
        base_node to form different edges.
    indices : list or array
        Set of nodes indices to be paired with base_node.
    distance : list or array
        Set of distances between all nodes in 'indices' and base_node.
    threshold : float
        Edge distance threshold. All edges with distance larger than
        'threshold' will not be added to G.

    """

    for c in np.arange(len(indices)):
        if distance[c] <= threshold:
            # If the distance between vertices is less than a given
            # threshold, add edge (i[0], i[c]) to Graph.
            G.add_weighted_edges_from([(base_node, indices[c],weights[c])])
            # G.nodes[base_node]["w_corner"] = points_df.iloc[base_node]["w_corner"]
            # G.nodes[indices[c]]["w_corner"] = points_df.iloc[indices[c]]["w_corner"]
## used to add all atts at one time
#             w_corners = dict(points_df.iloc[list(indices[c])]["w_corner"])
#             nx.set_node_attributes(G, name='w_corner', values=w_corners)



def array_to_graph(arr, G_weights, base_id, kpairs, knn, nbrs_threshold, nbrs_threshold_step, graph_threshold=np.inf, return_step=False):
    # Initializing graph.
    G = nx.Graph()
    # Generating array of all indices from 'arr' and all indices to process
    # 'idx'.
    idx_base = np.arange(arr.shape[0], dtype=int)
    idx = np.arange(arr.shape[0], dtype=int)
    # Initializing NearestNeighbors search and searching for all 'knn'
    # neighboring points arround each point in 'arr'.
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            leaf_size=15, n_jobs=-1).fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    indices = indices.astype(int)
    # Initializing variables for current ids being processed (current_idx)
    # and all ids already processed (processed_idx).
    current_idx = [base_id]
    processed_idx = [base_id]

    # Setting up the register of at which step each point was added to the
    # graph.
    step_register = np.full(arr.shape[0], np.nan)
    current_step = 0
    step_register[base_id] = current_step

    # Looping while there are still indices (idx) left to process.
    pbar = tqdm(total=arr.shape[0], unit = "B", unit_scale=True, position=0, leave=True)
    while idx.shape[0] > 0:

        # Increasing a single step count.
        current_step += 1

        # If current_idx is a list containing several indices.
        if len(current_idx) > 0:

            # Selecting NearestNeighbors indices and distances for current
            # indices being processed.
            nn = indices[current_idx]
            dd = distances[current_idx]
            ww = G_weights[current_idx]

            # Masking out indices already contained in processed_idx.
            mask1 = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)

            # Initializing temporary list of nearest neighbors. This list
            # is latter used to accumulate points that will be added to
            # processed points list.
            nntemp = []

            # Looping over current indices's set of nn points and selecting
            # knn points that hasn't been added/processed yet (mask1).
            for i, (n, d, w, g) in enumerate(zip(nn, dd,  ww, current_idx)):
                nn_idx = n[mask1[i]][0:kpairs+1]
                dd_idx = d[mask1[i]][0:kpairs+1]
                ww_idx = w[mask1[i]][0:kpairs+1]
                nntemp.append(nn_idx)
                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, ww_idx, graph_threshold)
            # Obtaining an unique array of points currently being processed.
            current_idx = np.unique([t2 for t1 in nntemp for t2 in t1])

        # If current_idx is an empty list.
        elif len(current_idx) == 0:

            # Getting NearestNeighbors indices and distance for all indices
            # that remain to be processed.
            idx2 = indices[idx]
            dist2 = distances[idx]

            # Masking indices in idx2 that have already been processed. The
            # idea is to connect remaining points to existing graph nodes.
            mask1 = np.in1d(idx2, processed_idx).reshape(idx2.shape)
            # Masking neighboring points that are withing threshold distance.
            mask2 = dist2 < nbrs_threshold
            # mask1 AND mask2. This will mask only indices that are part of
            # the graph and within threshold distance.
            mask = np.logical_and(mask1, mask2)

            # Getting unique array of indices that match the criteria from
            # mask1 and mask2.
            temp_idx = np.unique(np.where(mask)[0])
            # Assigns remaining indices (idx) matched in temp_idx to
            # current_idx.
            current_idx = idx[temp_idx]

            # Selecting NearestNeighbors indices and distances for current
            # indices being processed.
            nn = indices[current_idx]
            dd = distances[current_idx]
            ww = G_weights[current_idx]
            # Masking points in nn that have already been processed.
            # This is the oposite approach as above, where points that are
            # still not in the graph are desired. Now, to make sure the
            # continuity of the graph is kept, join current remaining indices
            # to indices already in G.
            mask = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)

            # Initializing temporary list of nearest neighbors. This list
            # is latter used to accumulate points that will be added to
            # processed points list.
            nntemp = []

            # Looping over current indices's set of nn points and selecting
            # knn points that have alreay been added/processed (mask).
            # Also, to ensure continuity over next iteration, select another
            # kpairs points from indices that haven't been processed (~mask).
            for i, (n, d, w, g) in enumerate(zip(nn, dd, ww, current_idx)):
                nn_idx = n[mask[i]][0:kpairs+1]
                dd_idx = d[mask[i]][0:kpairs+1]
                ww_idx = w[mask[i]][0:kpairs+1]

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, ww_idx, graph_threshold)

                nn_idx = n[~mask[i]][0:kpairs+1]
                dd_idx = d[~mask[i]][0:kpairs+1]
                ww_idx = w[~mask[i]][0:kpairs+1]

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, ww_idx, graph_threshold)

            # Check if current_idx is still empty. If so, increase the
            # nbrs_threshold to try to include more points in the next
            # iteration.
            if len(current_idx) == 0:
                nbrs_threshold += nbrs_threshold_step

        # Appending current_idx to processed_idx
        old_value = len(processed_idx)
        processed_idx = np.append(processed_idx, current_idx )
        processed_idx = np.unique(processed_idx).astype(int)
        new_value = len(processed_idx)
        pbar.update(new_value-old_value)
        # Generating list of remaining proints to process.
        idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

        # Adding new nodes to the step register.
        current_idx = np.array(current_idx).astype(int)
        step_register[current_idx] = current_step

    if return_step is True:
        return G, step_register
    else:
        return G
