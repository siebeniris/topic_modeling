deduplication_prompt = """
You are a data analysis expert tasked with deduplicating and reorganizing a hierarchical list of data clusters. Your goal is to create a more concise and representative set of high-level clusters, while retaining the essential information from the original structure. You are working with user behavior data, and your work will improve safety, monitoring, and observability.

**Input Data:**

The input is a hierarchical list of clusters. Each cluster has a name, a list of sub-clusters, and a justification explaining why those sub-clusters were grouped together to form the parent cluster.

{input}

Your task is to reduce the number of top-level clusters to approximately {desired_names}, while preserving the key information contained within the sub-clusters and justifications.

**Deduplication and Reorganization Process:**

Follow these steps to deduplicate and reorganize the cluster hierarchy:

1. **Analyze Existing Clusters:** Carefully read the name, list of sub-clusters, and justification for *each* top-level cluster. Understand the intended meaning and the rationale behind the existing grouping.
2. **Identify Redundancy and Overlap:** Compare the clusters to identify:
    * **Redundant Clusters:** Clusters that cover very similar topics with overlapping sub-clusters.
    * **Overlapping Sub-Clusters:** Sub-clusters that appear in multiple top-level clusters, suggesting a potential for reorganization.
    * **Hierarchical Inconsistencies:** Situations where a top-level cluster could reasonably be considered a sub-cluster of another.
3. **Reasoning for Merging/Reorganization:** For each potential merging or reorganization, perform the following reasoning:
    * **Identify Shared Themes:** What overarching theme connects the clusters being considered for merging?
    * **Evaluate Justifications:** How well do the existing justifications support or contradict the proposed merge?
    * **Assess Information Loss:** What information would be lost by merging these clusters? Can that information be retained in the new cluster name or justification?
    * **Consider Specificity vs. Generality:** Would the merged cluster be too general to be useful, or would it create a more coherent and meaningful category?
4. **Execute Merges/Reorganizations:** Based on your reasoning, perform the following actions:
    * **Merge Redundant Clusters:** Combine redundant clusters into a single cluster with a new name and justification that accurately reflects the combined content.
    * **Reassign Sub-Clusters:** If a sub-cluster appears in multiple top-level clusters, consider whether it should be moved to a higher-level cluster that encompasses both.
    * **Adjust Hierarchy:** If appropriate, move a top-level cluster to become a sub-cluster of another.
5. **Name and Justify New Clusters:** For any new clusters created through merging or reorganization, create a clear, concise name and a comprehensive justification that explains:
    * **The overarching theme of the new cluster.**
    * **Why the included sub-clusters belong together.**
    * **How the new cluster relates to the original clusters it replaced.**
6. **Final Review:** Review the final set of top-level clusters to ensure that:
    * **They are distinct from each other.**
    * **They collectively represent the range of topics covered by the original data.**
    * **The names and justifications are clear, accurate, and informative.**
    * **All original sub-clusters are accounted for within the new structure.**

7. Aim for approximately {desired_names} top-level clusters. However, the exact number is less important than the quality and representativeness of the final result. A reasonable range is {int(desired_names * 0.5)} to {int( desired_names * 1.5)}.

8. Do not hesitate to include clusters that describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful behavior are slightly preferred); specificity is necessary for effective monitoring and enforcement.

9. Prefer outputting specific cluster names over generic or vague ones, provided the names are still correct.

**Output Requirements:**

Present your final answer in the following format:

<deduplicated_hierarchy>
<cluster>
<name>[New Cluster Name]</name>
<justification>[Justification for the new cluster, explaining the merge/reorganization]</justification>
<sub_clusters>
<sub_cluster>[Sub-cluster Name]</sub_cluster>
<sub_cluster>[Sub-cluster Name]</sub_cluster>
...
</sub_clusters>
</cluster>
<cluster>
<name>[New Cluster Name]</name>
<justification>[Justification for the new cluster, explaining the merge/reorganization]</justification>
<sub_clusters>
<sub_cluster>[Sub-cluster Name]</sub_cluster>
<sub_cluster>[Sub-cluster Name]</sub_cluster>
...
</sub_clusters>
</cluster>
...
</deduplicated_hierarchy>

Before providing your final answer, use the `<scratchpad>` tags to detail your reasoning process for each merge, reorganization, or new cluster creation. Include the shared themes, evaluation of justifications, assessment of information loss, and considerations of specificity vs. generality.

Assistant: I understand. I will analyze the cluster hierarchy, identify areas for deduplication and reorganization, and create a more concise and representative set of top-level clusters, using the provided justifications to guide my decisions. My reasoning will be detailed in the `<scratchpad>`.

<scratchpad>

"""



updated_deduplication_prompt="""
You are a data analysis expert tasked with deduplicating and reorganizing a hierarchical list of data clusters. Your goal is to create a concise and well-structured set of non-overlapping hierarchical clusters, while retaining the essential information from the original structure. You are working with user behavior data, and your work will improve safety, monitoring, and observability.

**Input Data:**

The input is a hierarchical list of clusters. Each cluster has a name, a list of sub-clusters, and a justification explaining why those sub-clusters were grouped together to form the parent cluster.

{input}

**Deduplication and Reorganization Process:**

Follow these steps to deduplicate and reorganize the cluster hierarchy:

1. **Analyze Existing Clusters:** Carefully read the name, list of sub-clusters, and justification for *each* cluster at all levels of the hierarchy.  Understand the intended meaning and the rationale behind the existing groupings.

2. **Identify Redundancy, Overlap, and Hierarchical Inconsistencies:** Compare the clusters to identify:

    * **Redundant Clusters:** Clusters that cover very similar topics with highly overlapping sub-clusters.

    * **Overlapping Sub-Clusters:** Sub-clusters that appear in multiple parent clusters, suggesting a potential for reorganization to ensure non-overlapping clusters. Quantify the degree of overlap (e.g., percentage of sub-clusters shared).

    * **Hierarchical Inconsistencies:** Situations where a cluster at one level could reasonably be considered a sub-cluster of another at the same or different level, or where the hierarchical relationship seems illogical.

    * **Missing Clusters:** Identify potential high-level clusters that could provide a more coherent structure but are currently absent.

3. **Reasoning for Merging, Splitting, Reorganization, and Creation:** For each potential action (merging, splitting, reorganizing, creating a new cluster), perform the following reasoning:

    * **Identify Shared Themes:** What overarching theme connects the clusters being considered for merging, or what distinct themes justify splitting a cluster?

    * **Evaluate Justifications:** How well do the existing justifications support or contradict the proposed action? Are the justifications consistent and logically sound?

    * **Assess Information Loss/Gain:** What information would be lost or gained by performing this action? How can the action be performed to minimize information loss and maximize information gain?

    * **Consider Specificity vs. Generality:** Would the resulting cluster(s) be too general or too specific to be useful? Aim for a balance that provides meaningful categorization without obscuring important details.

    * **Assess Impact on Cluster Size:** How would the action impact the number of sub-clusters in each cluster? Strive for a balance in cluster sizes where possible.

    * **Evaluate Alternative Structures:** Are there alternative ways to restructure the hierarchy that might be more effective?

4. **Execute Merges, Splits, Reorganizations, and Creations:** Based on your reasoning, perform the following actions:

    * **Merge Redundant Clusters:** Combine redundant clusters into a single cluster with a new name and justification that accurately reflects the combined content.  Ensure the new justification addresses all aspects of the original clusters.

    * **Split Overly Broad Clusters:** If a cluster is too general and encompasses disparate sub-themes, split it into multiple more specific clusters.

    * **Reassign Sub-Clusters:** If a sub-cluster appears in multiple parent clusters, determine the most appropriate parent based on thematic coherence and justification.  Ensure the sub-cluster resides in only one parent to achieve non-overlapping clusters.

    * **Adjust Hierarchy:** Move clusters to different levels of the hierarchy to reflect more accurate relationships. Create new intermediate-level clusters as needed to bridge gaps.

    * **Create New Clusters:**  If the existing hierarchy lacks a crucial high-level categorization, create a new cluster to encompass related sub-clusters.

5. **Name and Justify New and Modified Clusters:** For any new clusters created or existing clusters significantly modified through merging, splitting, or reorganization, create a clear, concise name and a comprehensive justification that explains:

    * **The overarching theme of the cluster.**
    * **Why the included sub-clusters belong together (or why the cluster was split).**
    * **How the cluster relates to the original clusters it replaced or modified.**
    * **How the new structure improves upon the original in terms of clarity, coherence, and non-overlap.**

6. **Final Review:** Review the final set of clusters at all levels of the hierarchy to ensure that:

    * **They are as distinct from each other as possible (minimize overlap).**
    * **They collectively represent the range of topics covered by the original data.**
    * **The names and justifications are clear, accurate, and informative.**
    * **All original sub-clusters are accounted for within the new structure.**
    * **The hierarchy is logically sound and reflects meaningful relationships between clusters.**
    * **The level of granularity is appropriate for the intended use (safety, monitoring, observability).**

7. **Prioritize Specificity and Actionability:** Do not hesitate to include clusters that describe socially harmful or sensitive topics; specificity is necessary for effective monitoring and enforcement.  Prefer specific cluster names over generic ones, provided the names are still accurate and representative. The ultimate goal is to create a hierarchy that enables clear identification and targeted action.

**Output Requirements:**

Present your final answer in the following XML format:

```xml
<deduplicated_hierarchy>
  <cluster>
    <name>[New Cluster Name]</name>
    <justification>[Justification for the new cluster, explaining the merge/split/reorganization, and why this structure is superior]</justification>
    <sub_clusters>
      <sub_cluster>[Sub-cluster Name]</sub_cluster>
      <sub_cluster>[Sub-cluster Name]</sub_cluster>
      ...
    </sub_clusters>
  </cluster>
  <cluster>
    <name>[New Cluster Name]</name>
    <justification>[Justification for the new cluster, explaining the merge/split/reorganization, and why this structure is superior]</justification>
    <sub_clusters>
      <sub_cluster>[Sub-cluster Name]</sub_cluster>
      <sub_cluster>[Sub-cluster Name]</sub_cluster>
      ...
    </sub_clusters>
  </cluster>
  ...
</deduplicated_hierarchy>
```

Before providing your final answer, use the <scratchpad> tags to detail your reasoning process for each merge, split, reorganization, or new cluster creation. Include the shared themes, evaluation of justifications, assessment of information loss/gain, considerations of specificity vs. generality, impact on cluster size, and evaluation of alternative structures. For overlapping sub-clusters, quantify the degree of overlap.

Assistant: I understand. I will analyze the cluster hierarchy, identify areas for deduplication, splitting, and reorganization, and create a more concise and representative set of clusters, using the provided justifications to guide my decisions and creating new justifications where necessary. I will prioritize non-overlapping clusters and a logically sound hierarchy. My reasoning will be detailed in the <scratchpad>.
"""