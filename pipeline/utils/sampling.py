import numpy as np
from sklearn.metrics import pairwise_distances



def max_min_sampling(embeddings, sample_size, metric='euclidean', random_state=42):
    """
    Selects a subset of embeddings using the Max-Min algorithm.
    """
    np.random.seed(random_state)

    # 1. Select the first point randomly
    initial_index = np.random.choice(len(embeddings))
    sample_indices = [initial_index]
    sample = [embeddings[initial_index]]

    # 2. Iterate until the desired sample size is reached
    while len(sample) < sample_size:
        # 3. Calculate the minimum distance from each remaining point
        #    to the points already in the sample
        distances = pairwise_distances(embeddings, sample, metric=metric)
        min_distances = np.min(distances, axis=1)

        # 4. Select the point with the maximum minimum distance
        next_index = np.argmax(min_distances)
        sample_indices.append(next_index)
        sample.append(embeddings[next_index])

    # 5. Convert the sample to a NumPy array
    sample = np.array(sample)
    return sample, sample_indices


def stratified_max_min_sampling_proportional(embeddings, labels, total_sample_size, metric='euclidean', random_state=42):

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_proportions = label_counts / len(labels)
    sampled_embeddings = []
    sampled_labels = []
    sampled_indices = []

    for label, proportion in zip(unique_labels, label_proportions):
        # 1. Calculate the sample size for the current label
        sample_size_for_label = int(round(total_sample_size * proportion))
        #Ensure sample size for label is at least 1, if total_sample_size is relatively small
        if sample_size_for_label == 0 and total_sample_size > 0 :
            sample_size_for_label = 1

        # 2. Get the embeddings and indices for the current label
        label_indices = np.where(labels == label)[0]
        label_embeddings = embeddings[label_indices]

        # 3. Perform Max-Min sampling on the embeddings for the current label
        if len(label_embeddings) > 0: #Check if there's any embeddings for that particular label
            if len(label_embeddings) < sample_size_for_label: #If label embedding size is less than sample size, then just take all of it.
                max_min_sample, max_min_indices = label_embeddings, list(range(len(label_embeddings)))
            else:
                max_min_sample, max_min_indices = max_min_sampling(
                    label_embeddings, sample_size_for_label, metric=metric, random_state=random_state
                )
            # 4. Store the sampled embeddings, labels, and indices
            sampled_embeddings.extend(max_min_sample)
            sampled_labels.extend([label] * len(max_min_sample))  # Assign the correct label
            #Need to revert it back to the original label index
            sampled_indices.extend(label_indices[max_min_indices].tolist())
        else:
            print(f"Warning: No embeddings found for label {label}")

    # 5. Convert the sampled embeddings and labels to NumPy arrays
    sampled_embeddings = np.array(sampled_embeddings)
    sampled_labels = np.array(sampled_labels)
    sampled_indices = np.array(sampled_indices)

    return sampled_embeddings, sampled_labels, sampled_indices