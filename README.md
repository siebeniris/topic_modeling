# Organized Pipeline for Topic modeling 

### Summarization using LLMs

- `summarizer.py`

- Points
    * using few-shot prompting
    * models :
        - GPT-4.1-nano  (18 hours, sequentially )
        - GPT-4o-mini (36 hours)
        - Claude-3.5-haiku
        - Llama3-8b (GPU needed.)
    * evaluation:
        - cross-examing the output with string matching metrics
        - check if the relevance and conversation continuation is at least partially correct 
            - NA vs. non-NA
        
        - using SAE/NER model to detect PII
            - GPT-4.1-nano gives out very specific details in summarization


- `eval/eval_summarization.py`

### Extract the embeddings

`extract_embeddings.py`

### Clustering 

- Using the `task` and `subject` output from LLMs to run clustering directly. 


# Hyperparameter tuning

```
python -m pipeline.clusterer --mode exp 
```


# Clustering 

Do grid search first to define the hyper parameters min_clusters_size and min_samples_size. And then to run the clustering directly.

For subject embeddings. 
```
python -m pipeline.clusterer --data_path data/gpt-4o-mini_20250508_161651 --content subject --mode clustering --min_clusters_size 300 --min_samples_size 5
```

* For task embeddings
```
python -m pipeline.clusterer --data_path data/gpt-4o-mini_20250508_161651 --content task --mode clustering --min_clusters_size 150 --min_samples_size 5
```


### Output files

- from task clustering
    - `topicmodelingpipeline/data/clustering/task_clusters.csv`

- cluster ids.
    - `topicmodelingpipeline/data/clustering/task_clusters_overall.csv`


### Naming base clusters and Hierarchization 

```
naming_base_clusters.py

merge_clusters.py

deduplicate_clusters.py
```

### Annotation and Post-processing

After we have the summaries, the sub-clusters and higher_level clusters, conduct human evaluation. 

The evaluation and annotation outcome are in the `data` folder.

Follow the annotation guideline. 

Evaluation of annotations are done. 


### KNN Classifier

* Data and output. 

```
Data
- knn_data 
    - sub_cluster
    - higher_level_cluster 
    train and test data in embeddings and numpy array of labels.

Output
- standard scaler (based on train data)
- umap reducer (based on train data)
- knn classifier for higher level clusters 
- knn classifier for sub clusters
```

* Grid search for the best hyperparameters
`KNN_grid_search.py`

* Preprocessing the data and get the scaler and reducer
`KNN_data_preprocessing.py`

* Experiment incorporating new data with sampling and evaluating KNN
- the same for higher-level clusters
`new_input_KNN_subcluster.py`


