import numpy as np
import time
from query_graph_kgc_se import QueryGraph

def star_to_triples(query_parts, dict_join, store_join_ids, var_id, joins, special_format = False):
    """
    Star query converted to triples
    :param query_parts: actual star query of for s-p1-o1-p2-o2
    :param dict_join: stores the position which will be joined with the var for joining
    :param store_join_ids: which var needs to be stored
    :param var_id: if the query is a complex query here we will store the actual variable ?id counter from which we need to continue
    :return:
    """
    store_var_dict = dict()
    query_triples = []
    if not special_format:
        pattern_parts = query_parts.split("-")
    else:
        pattern_parts = ["*"]
        for part in query_parts.split(","):
            for p in part.split("-"):
                pattern_parts.append(p)

    s = pattern_parts[0]
    s_var = s

    ids = list()
    for index in range(joins):
        if index == 0:
            ids.append(3)
        else:
            ids.append(3 + (3 * (index)))

    # Get the ids from the pattern parts
    sims = list()
    for i in ids:
        sims.append(pattern_parts[i])

    # loop ids in reverse order
    ids = ids[::-1]
    for i in ids:
        pattern_parts.pop(i)
    
    if 0 in dict_join:
        s_var = dict_join[0]
    elif "*" in s_var or "?" in s_var:
        s_var = "?" + str(var_id)
        var_id += 1

    if 0 in store_join_ids:
        store_var_dict[0] = s_var

    i = 1
    times = (len(pattern_parts) - 1) /2
    # len_parts = len(pattern_parts)
    while times > 0:
        #print(str(i) + "/" + str(times))
        p = pattern_parts[i]
        o = pattern_parts[i + 1]

        if (i + 1) in dict_join:
            o_var = dict_join[(i + 1)]
        else:
            o_var, var_id = create_var_id(o, var_id)

        if (i + 1) in store_join_ids:
            store_var_dict[i + 1] = o_var

        if i in dict_join:
            p_var = dict_join[i]
        else:
            p_var, var_id = create_var_id(p, var_id)

        if i in store_join_ids:
            store_var_dict[i] = p_var

        query_triples.append([s_var, p_var, o_var])
        i += 2
        times -= 1

    return query_triples, store_var_dict, var_id, sims


def chain_to_triples(query_parts, dict_join, store_join_ids, var_id, joins, special_format = False):
    """
    Chain query converted to triples
    :param query_parts: actual chain query of for s-p1-o1-p2-o2
    :param dict_join: stores the position which will be joined with the var for joining
    :param store_join_ids: which var needs to be stored
    :param var_id: if the query is a complex query here we will store the actual variable ?id counter from which we need to continue
    :return:
    """

    store_var_dict = dict()
    query_triples = []
    pattern_parts = query_parts.split("-")

    s = pattern_parts[0]
    s_var = s

    ids = list()
    for index in range(joins):
        if index == 0:
            ids.append(2)
        else:
            ids.append(3 + (3 * (index)))

    # Get the ids from the pattern parts
    sims = list()
    for i in ids:
        sims.append(pattern_parts[i])

    # loop ids in reverse order
    ids = ids[::-1]
    for i in ids:
        pattern_parts.pop(i)

    if 0 in dict_join:
        s_var = dict_join[0]
    elif "*" in s_var or "?" in s_var:
        s_var = "?" + str(var_id)
        var_id += 1
    if 0 in store_join_ids:
        store_var_dict[0] = s_var

    i = 1
    times = (len(pattern_parts) - 1) /2
    # len_parts = len(pattern_parts)
    firstTime = True
    while times > 0:
        # print(str(i) + "/" + str(times))
        p = pattern_parts[i]
        o = pattern_parts[i + 1]

        if not firstTime:
            s_var = query_triples[len(query_triples) - 1][2]

        if (i + 1) in dict_join:
            o_var = dict_join[i + 1]
        else:
            o_var, var_id = create_var_id(o, var_id)

        if (i + 1) in store_join_ids:
            store_var_dict[i + 1] = o_var

        if i in dict_join:
            p_var = dict_join[i]
        else:
            p_var, var_id = create_var_id(p, var_id)

        if i in store_join_ids:
            store_var_dict[i] = p_var

        query_triples.append([s_var, p_var, o_var])
        i += 2
        times -= 1
        firstTime = False

    return query_triples, store_var_dict, var_id, sims


def create_var_id(var, var_id):
    if "*" in var or "?" in var:
        var = "?" + str(var_id)
        var_id += 1
    return var, var_id



def read_complex_queries(files, d, b, n, e, limit = 100_000_000):
    """
    Processes the example files
    :param files: the file contains two queries joined by the last and the first attribute
    :param d: int, the number of distinct nodes (subjects + objects) in KG
    :param b: int, the number of distinct edges (predicates) in KG
    :param n: int, the number of nodes in the subgraph
    :param e: int, the number of edges in the subgraph
    :param limit: max queries to read
    :return:
    """
    time_start = time.time()
    X, A, E, y = [], [], [], []

    # The generated join queries are with the last and the first attribute
    join_spec = ("last", "first")
    join_specs = [[join_spec] for i in range(len(files))]
    # print(join_specs)

    # This is a join between two already complex queries of types (star, chain)
    for i, file in enumerate(files):
        j_s = join_specs[i]
        with open(file, "r") as f:
            for line_nb, line in enumerate(f):
                if (line_nb % 100000) == 0:
                    print("Reading " + file + ", line " + str(line_nb))
                if line_nb > limit:
                    break
                queries, card = line.split(":")
                card = int(card)
                query_parts1, query_parts2 = queries.split(",")

                # Here read all of the lines and merge the query triples and create the graph
                query_triples = []
                join_pairs = []
                # which ids need to be joined
                for j in j_s:
                    left_join_idx = 0 if j[0] == "first" else (len(query_parts1.split("-")) - 1)
                    right_join_idx = 0 if j[1] == "first" else (len(query_parts2.split("-")) - 1)
                    join_pairs.append((left_join_idx, right_join_idx))
                # print("The join pairs are " + str(join_pairs))

                store_join_ids = set()
                for j_p in join_pairs:
                    idx = j_p[0]
                    store_join_ids.add(idx)
                # print("Store join ids is " + str(store_join_ids))

                # Processing of query 1
                # query_parts1 = "*-1-*-2-*-2-*"
                dict_join = dict()
                var_id = 0
                query_triples1, store_var_dict, var_id = star_to_triples(query_parts1, dict_join, store_join_ids, var_id)
                # print("Query triple 1: " + str(query_triples1))
                query_triples.extend(query_triples1)

                dict_join = dict()
                # ovde beshe pair
                for j_p in join_pairs:
                    idx1 = j_p[0]
                    idx2 = j_p[1]
                    # print(store_var_dict)
                    dict_join[idx2] = store_var_dict[idx1]

                # Processing of query 2
                query_triples2, store_var_dict, var_id = chain_to_triples(query_parts2, dict_join, set(), var_id)
                # print("Query triple 2: " + str(query_triples2))

                query_triples.extend(query_triples2)
                # print("All query triples: " + str(query_triples))

                # For every query we create the graph
                graph = QueryGraph(d,b,n,e)
                graph.cardinality = card
                for query_triple in query_triples:
                    graph.add_triple(query_triple)
                # graph.print()
                x, ep, a, cardinality = graph.create_graph()

                X.append(x)
                E.append(ep)
                A.append(a)
                y.append(cardinality)

    end_time = time.time() - time_start
    return np.array(X), np.array(A), np.array(E), np.array(y), end_time


def read_queries(file, d, b, n, e, joins, query_type = "star", limit = 100_000_000, model = None):
    
    # Processes the example files
    # :param files: the file contains two queries joined by the last and the first attribute
    # :param d: int, the number of distinct nodes (subjects + objects) in KG
    # :param b: int, the number of distinct edges (predicates) in KG
    # :param n: int, the number of nodes in the subgraph
    # :param e: int, the number of edges in the subgraph
    # :param limit: max queries to read
    # :return:
    
    time_start = time.time()
    X, A, E, T, y, s = [], [], [], [], [], []

    # This is a join between two already complex queries of types (star, chain)
    with open(file, "r") as f:
        for line_nb, line in enumerate(f):
            if (line_nb % 100000) == 0:
                print("Reading " + file + ", line " + str(line_nb))
            if line_nb > limit:
                break
            if query_type == "star":
                queries, card = line.split(":")
            else:
                queries, card = line.split(",")
            card = int(card)

            # Here read all of the lines and merge the query triples and create the graph
            query_triples = []
            dict_join = dict()
            var_id = 0

            if query_type == "star":
                query_triples1, store_var_dict, var_id, sims = star_to_triples(queries, dict_join, set(), var_id, joins, special_format=True)
                # print("Query triple 1: " + str(query_triples1))
                query_triples.extend(query_triples1)
            elif query_type == "chain":
                query_triples2, store_var_dict, var_id, sims = chain_to_triples(queries, dict_join, set(), var_id, joins)
                # print("Query triple 2: " + str(query_triples2))
                query_triples.extend(query_triples2)
            # print("All query triples: " + str(query_triples))

            # For every query we create the graph
            x, ep, a = [], [], []
            try:
                for query_triple in query_triples:
                    graph = QueryGraph(d,b,n,e)
                    graph.cardinality = card
                    graph.add_triple(query_triple)
                    x1, ep1, a1, cardinality = graph.create_graph()
                    x.append(x1)
                    ep.append(ep1)
                    a.append(a1)    
            except:
                continue    

            X.append(x)
            E.append(ep)
            A.append(a)
            y.append(cardinality)
            s.append(sims)

    end_time = time.time() - time_start
    return np.array(X), np.array(A), np.array(E), np.array(s), np.array(y), end_time



def read_combined(d, b, n, e, file_name_star, file_name_chain, train_tuples = 10000, test_mode = "star"):
    """
    Creates encoding from queries in the file
    :param d: int, the number of distinct nodes (subjects + objects) in KG
    :param b: int, the number of distinct edges (predicates) in KG
    :param n: int, the number of nodes in the subgraph
    :param e: int, the number of edges in the subgraph
    :param file_name_star: str, the input file name of the queries used for training or testing
    :param file_name_chain: str, the input file name of the queries used for training or testing
    :param train_tuples: int, limits the training sample size, in default we train on all of them
    :param test_mode: str, it can be star or chain
    :return: encoded queries
    """
    if "star" in test_mode:
        return read_queries(file_name_star, d, b, n, e, query_type="star")
    if "chain" in test_mode:
        return read_queries(file_name_chain, d, b, n, e, query_type="chain")

    X_s, A_s, E_s, y_s, time_start_star = read_queries(file_name_star, d, b, n, e, query_type="star", limit = train_tuples)
    X_c, A_c, E_c, y_c, time_start_chain = read_queries(file_name_chain, d, b, n, e, query_type="chain", limit = train_tuples)

    X = np.concatenate([X_s, X_c], axis = 0)
    A = np.concatenate([A_s, A_c], axis = 0)
    E = np.concatenate([E_s, E_c], axis = 0)
    y = np.concatenate([y_s, y_c], axis = 0)

    return X, A, E, y, (time_start_star + time_start_chain)


def read_combined_all_sizes_star_or_chain(d, b, n, e, file_names, train_tuples = 10000, model = None):
    
    # Creates encoding from queries in the file
    # :param d: int, the number of distinct nodes (subjects + objects) in KG
    # :param b: int, the number of distinct edges (predicates) in KG
    # :param n: int, the number of nodes in the subgraph
    # :param e: int, the number of edges in the subgraph
    # :param file_names: str, the input file name of the queries used for training or testing
    # :param train_tuples: int, limits the training sample size, in default we train on all of them
    # :return: encoded queries
    
    all_X = []
    all_A = []
    all_E = []
    all_y = []
    time = 0
    for i in range(len(file_names)):
        if 'star' in file_names[i]:
            X_i, A_i, E_i, y_i, time_start_i = read_queries(file_names[i], d, b, n, e, query_type="star", limit = train_tuples, model = model)
        else:
            X_i, A_i, E_i, y_i, time_start_i = read_queries(file_names[i], d, b, n, e, query_type="chain", limit = train_tuples, model = model)
        all_X.append(X_i)
        all_A.append(A_i)
        all_E.append(E_i)
        all_T.append(T_i)
        all_y.append(y_i)
        time += time_start_i

    X = np.concatenate(all_X, axis = 0)
    A = np.concatenate(all_A, axis = 0)
    E = np.concatenate(all_E, axis = 0)
    y = np.concatenate(all_y, axis = 0)

    return X, A, E, y, time