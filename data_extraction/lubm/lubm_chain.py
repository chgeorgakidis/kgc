from multiprocessing import Process
from neo4j import GraphDatabase
import pandas as pd
import random
import os
import gc
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run Cypher queries in parallel
def run_query(tx, query):
    return tx.run(query)

similarities = [0.2, 0.4, 0.6, 0.8, 1.0]

# Process that runs in parallel, which generates a number of queries and writes them to a file
def run_process(i):
    my_graph = GraphDatabase.driver('bolt://<graph_address>', auth=('xxxx', 'xxxx'))
    count = 0
    while True:
        cardinality = 0
        with open(filename, 'a') as data_file:

            # Randomly select one node
            curr_node_idx = random.randint(0, len(node_list)-1)
            curr_node = node_list[curr_node_idx]

            # Randomly select a query from node dict for this node
            el = random.choice(list(node_dict[curr_node]))

            # Split the elements
            els_split = el.split('!;;!')

            x_ids = list()
            rels = list()
            literals = list()
            for i in range(joins):
                rels.append(str(els_split[(i) + (i*2)]))
                x_ids.append(str(els_split[(i+1) + (i*2)]))
                literals.append(str(els_split[(i+2) + (i*2)]))

            # Randomly select joins similarities
            sims = list()
            for i in range(joins):
                sims.append(similarities[random.randint(0, len(similarities)-1)])

            no_unbound = random.randint(0, max_no_unbound)
            query_cypher = 'MATCH '
            for i in range(joins):
                query_cypher += '(x' + str(i+1) + ')-[:' + rels[i] + ']->'
            query_cypher += '(x' + str(joins+1) + ')' + ' WHERE '
            for i in range(joins):
                idx = i+1
                if idx > 1:
                    idx = i+2
                query_cypher += '(apoc.text.levenshteinSimilarity(x' + str(idx) + '.ns0__name, \"'  + str(literals[i]) + '\")) >= ' + str(sims[i]) + ' AND '
            query_cypher = query_cypher[:-4]
            query_cypher += ' RETURN COUNT(*)'

            unbound_pos = list()
            if no_unbound != 0:
                # Randomly select no_unbound positions. Then split the query, remove the corresponding parts and re-join everything    
                for i in range(no_unbound):
                    unbound_pos.append(random.randint(0, joins-1))
                unbound_pos = list(set(unbound_pos))
                unbound_pos.sort()
                unbound_pos = unbound_pos[::-1]

                query_cypher_split = query_cypher.split('WHERE')
                query_cypher_split2 = query_cypher_split[1].split('RETURN')
                query_cypher_split3 = query_cypher_split2[0].split('AND')
                # Remove the unbound positions
                for index in unbound_pos:
                    del query_cypher_split3[index]
                query_cypher = query_cypher_split[0] + 'WHERE'
                for el in query_cypher_split3:
                    query_cypher += el + 'AND'
                query_cypher = query_cypher[:-3] + 'RETURN' + query_cypher_split2[1]

            try:
                with my_graph.session() as session:
                    result = session.run(query_cypher)
                    res = list(result)
                    cardinality = res[0][0]
            except:
                continue

            query = str(x_ids[0]) + '-' + str(rel_dict[rels[0]]) + '-' + str(sims[0]) + '-'
            for i in range(joins-1):
                query += str(rel_dict[rels[i+1]]) + '-' + str(x_ids[i+1]) + '-' + str(sims[i+1]) + '-'
            query = query[:-1]

            query_split = query.split('-')
            if no_unbound != 0:
                for index in unbound_pos:
                    if index == 0:
                        query_split[0] = '*'
                        query_split[2] = '1.0'
                    else:
                        query_split[4 + (3*(index - 1))] = '*'
                        query_split[4 + (3*(index - 1)) + 1] = '1.0'

            query_split.insert(3, '*')
            query = '-'.join(query_split) + ',' + str(cardinality)

            data_file.write(query + '\n')
        data_file.close()
        
        if count*i % 10 == 0:
            print('Done ' + str(count*i))
            gc.collect()
        count += 1

# Main function
if __name__ == '__main__':

    # Get first argument
    joins = int(sys.argv[1])

    max_no_unbound = joins - 1

    # Read nodes and names from CSV file in a dictionary
    df = pd.read_csv('Data/nodes_with_labels_chain_' + str(joins) + '.csv', header=None, sep='!__!')

    node_dict = {}
    node_list = df[0].tolist()
    data_list = df[1].tolist()

    for i in range(len(node_list)):
        if node_list[i] not in node_dict:
            node_dict[node_list[i]] = list()
            node_dict[node_list[i]].append(data_list[i])
        else:
            node_dict[node_list[i]].append(data_list[i])

    print('Done reading nodes')

    # Read relationships in a dictionary
    df = pd.read_csv('Data/relationships.csv', header=None, sep='!__!')

    rel_dict = {}
    rel_list = df[0].tolist()

    for i in range(len(rel_list)):
        rel_dict[rel_list[i]] = i

    # Filename to write the queries to
    filename = 'training_lubm_chain_super_sim_' + str(joins) + '.csv'

    # Number of processes to run in parallel
    num_processes = 12

    # Create a list of processes
    processes = []

    # Create a process for each process
    for i in range(num_processes):
        processes.append(Process(target=run_process, args=(i+1,)))

    # Start the processes
    for i in range(num_processes):
        processes[i].start()

    # Join the processes
    for i in range(num_processes):
        processes[i].join()