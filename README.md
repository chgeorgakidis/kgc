# KGC
This repository contains the code and example data for the paper "Supervised Cardinality Estimation for Similarity Queries on Knowledge Graphs". 

Additional artifacts can be found as follows:
- LUBM: https://swat.cse.lehigh.edu/projects/lubm/
- WIKI: https://github.com/MillenniumDB/WDBench
- YELP: https://www.yelp.com/dataset
- SWDF: hhttp://www.scholarlydata.org
- LMKG code can be found [here](https://git.cs.uni-kl.de/dbis-public/lmkg/lmkg-learned-models-for-cardinality-estimation-in-knowledge-graphs/-/tree/master/lmkg-s) 

A subset of the training and testing data for the LUBM dataset is provided. To extract training data for the rest of the datasets, the corresponding neo4j knowledge graph must be built. Then the scripts in the `data_extraction` folder may be used.

The code was developed using Python 3.10.4, keras 2.7.0 and tensorflow 2.7.1.

To run the script for the star similarity query example, please issue the following command:

```bash
python kgc_exp.py star 5 lubm 1 100 128
```

For the chain similarity query:

```bash
python kgc_exp.py chain 2 lubm 1 100 128
```

For more information regarding the arguments of the script, please refer to the `kgc_exp.py` file. When the code is executed, the trained models and results are stored in the `models` and `results` folders respectively. After execution, the user may find the set of results under a subfolder named according to the used dataset.