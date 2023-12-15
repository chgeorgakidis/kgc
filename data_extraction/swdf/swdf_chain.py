from multiprocessing import Process
from neo4j import GraphDatabase
import pandas as pd
import random
import os
import gc

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
            el_split = el.split('!;;!')

            x1_id = str(el_split[1])
            x2_id = str(el_split[4])
            rel1 = str(el_split[0])
            rel2 = str(el_split[3])

            # Randomly select two similarities
            sim1 = similarities[random.randint(0, len(similarities)-1)]
            sim2 = similarities[random.randint(0, len(similarities)-1)]

            no_unbound = random.randint(0, max_no_unbound)
            if no_unbound == 1:
                unbound_pos = random.randint(0, 1)
                if unbound_pos == 0:
                    query_cypher = 'MATCH (x1)-[:' + rel1 + ']->(x2)-[:' + rel2 + ']->(x3) WHERE (apoc.text.levenshteinSimilarity(x3.rdfs__label, \"' + str(el_split[5]) + '\")) >= ' + str(sim2) + ' RETURN COUNT(*)'
                else:
                    query_cypher = 'MATCH (x1)-[:' + rel1 + ']->(x2)-[:' + rel2 + ']->(x3) WHERE (apoc.text.levenshteinSimilarity(x1.rdfs__label, \"' + str(el_split[2]) + '\")) >= ' + str(sim1) + ' RETURN COUNT(*)'
            else:
                query_cypher = 'MATCH (x1)-[:' + rel1 + ']->(x2)-[:' + rel2 + ']->(x3) WHERE (apoc.text.levenshteinSimilarity(x1.rdfs__label, \"' + str(el_split[2]) + '\")) >= ' + str(sim1) +  ' AND (apoc.text.levenshteinSimilarity(x3.rdfs__label, \"' + str(el_split[5]) + '\")) >= ' + str(sim2) + ' RETURN COUNT(*)'

            try:
                with my_graph.session() as session:
                    result = session.run(query_cypher)
                    res = list(result)
                    cardinality = res[0][0]
            except:
                continue

            if no_unbound == 0:
                query = x1_id + '-' + str(rel_dict[rel1]) + '-' + str(sim1) + '-*-' + str(rel_dict[rel2]) + '-' + x2_id + '-' + str(sim2) + ',' + str(cardinality)
            elif no_unbound == 1:
                if unbound_pos == 0:
                    query = '*' + '-' + str(rel_dict[rel1]) + '-1.0' + '-*-' + str(rel_dict[rel2]) + '-' + x2_id + '-' + str(sim2) + ',' + str(cardinality)
                elif unbound_pos == 1:
                    query = x1_id + '-' + str(rel_dict[rel1]) + '-' + str(sim1) + '-*-' + str(rel_dict[rel2]) + '-*' + '-' + str(sim2) + ',' + str(cardinality)

            data_file.write(query + '\n')
        data_file.close()
        
        if count*i % 10 == 0:
            print('Done ' + str(count*i))
            gc.collect()
        count += 1

# Main function
if __name__ == '__main__':

    joins = 2

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

    max_no_unbound = 1
    # Filename to write the queries to
    filename = 'training_swdf_chain_super_sim.csv'

    # Number of processes to run in parallel
    num_processes = 24

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