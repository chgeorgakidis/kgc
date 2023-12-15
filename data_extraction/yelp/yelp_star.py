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

            if len(node_dict[curr_node]) < joins:
                continue

            els = list()
            for i in range(joins):
                els.append(random.choice(list(node_dict[curr_node])))

            # If any of the elements is the same with another, continue
            if len(set(els)) < joins:
                continue

            # Split the elements
            els_split = list()
            for el in els:
                els_split.append(el.split('!;;!'))
            
            x_ids = list()
            rels = list()
            for el in els_split:
                x_ids.append(str(el[1]))
                rels.append(str(el[0]))

            # Randomly select joins similarities
            sims = list()
            for i in range(joins):
                sims.append(similarities[random.randint(0, len(similarities)-1)])
            
            no_unbound = random.randint(0, max_no_unbound)            
            query_cypher = 'MATCH '
            for i in range(joins):
                query_cypher += '(x1)-[:' + rels[i] + ']->(x' + str(i+2) + '), '
            query_cypher = query_cypher[:-2]
            query_cypher += ' WHERE '
            for i in range(joins):
                query_cypher += '(apoc.text.levenshteinSimilarity(x' + str(i+2) + '.text, \"' + str(els_split[i][2]) + '\")) >= ' + str(sims[i]) + ' AND '
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

            query = ''
            for i in range(joins):
                if i in unbound_pos:
                    query += str(rel_dict[rels[i]]) + '-*-1.0' + ','
                else:
                    query += str(rel_dict[rels[i]]) + '-' + x_ids[i] + '-' + str(sims[i]) + ','
            query = query[:-1] + ':' + str(cardinality)

            data_file.write(query + '\n')
        data_file.close()
        
        if count*i % 10 == 0:
            print('Done ' + str(count*i))
            gc.collect()
        count += 1

# Main function
if __name__ == '__main__':

    joins = int(sys.argv[1])
    max_no_unbound = joins - 1

    # Read nodes and names from CSV file in a dictionary
    df = pd.read_csv('Data/nodes_with_labels_users.csv', header=None, sep='!__!')

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
    filename = 'training_yelp_star_super_sim_' + str(joins) + '.csv'
    
    # Number of processes to run in parallel
    num_processes = 8

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