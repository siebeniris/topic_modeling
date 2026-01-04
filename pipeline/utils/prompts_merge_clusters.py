merge_clusters_template="""
You are tasked with creating higher-level cluster names based on a given list of clusters and their descriptions. Your goal is to identify broader, overarching categories that can effectively group one or more of the provided clusters. These higher-level categories should reflect shared themes, behaviors, or characteristics across the clusters, while maintaining clarity, specificity, and distinctiveness.

Steps to Follow:
1. Analyze the Clusters: Carefully review the provided clusters and their descriptions to identify recurring themes, patterns, or relationships. Look for shared characteristics or overarching ideas that connect multiple clusters.
2. Group Related Clusters: Organize clusters into logical groups based on their similarities. Each group should represent a cohesive theme or topic.
3. Create Higher-Level Cluster Names: Develop concise, descriptive names for the overarching categories. These names should capture the essence of the grouped clusters without being overly broad or vague.

Guidelines for Higher-Level Cluster Names:
1. Thematic Consistency: Each higher-level cluster name should represent a clear and cohesive theme that ties together the clusters it encompasses.
2. Specificity and Clarity: Avoid overly generic terms. The names should be specific enough to provide insight into the nature of the grouped clusters while remaining broad enough to encompass multiple clusters.
3. Distinctiveness: Ensure that each higher-level cluster name is unique and does not overlap significantly with others.
4. Alignment with Context: Consider the purpose of organizing the data (e.g., improving safety, monitoring, and observability) and ensure the names align with this goal.
5. Sensitivity to Content: If clusters involve socially harmful or sensitive topics, address them directly but thoughtfully, ensuring the names remain professional and neutral.

Output Requirements:
- Provide at least {low_bound_desired_number} and at most {upper_bound_desired_number} higher-level cluster names, with a target of {desired_number} names. You can generate more or less than {desired_number} names if you feel that more or fewer are appropriate and accurately capture the clusters.
- Clearly specify which clusters are associated with each higher-level cluster name.
- Use clear, concise language for the names.
- Present your final list in the specified format.

Cluster List:
<cluster_list> 
{cluster_list}
</cluster_list>  

Scratchpad for Brainstorming:
Before finalizing your list, use the scratchpad to:
- Analyze the clusters and identify common themes or patterns.
- Brainstorm potential higher-level cluster names.
- Refine your ideas to ensure they meet the guidelines.

Scratchpad Format:

[Use this space to analyze the clusters, identify common themes, and brainstorm potential higher-level cluster names. Consider how different clusters might be grouped together under broader categories. Limit this section to a paragraph or two.]

Final Output Format:
1. **[First Higher-Level Cluster Name]** **Associated Clusters:** [List of associated clusters]
[Second Higher-Level Cluster Name]
Associated Clusters: [List of associated clusters]

[Third Higher-Level Cluster Name]
Associated Clusters: [List of associated clusters]

...

{desired_number}. [Last Higher-Level Cluster Name]
Associated Clusters: [List of associated clusters]
"""



template_updated="""
You are an expert at identifying overarching themes and generating descriptive names for data clusters.

**Task:** Analyze the provided list of data clusters and their descriptions to identify broader, higher-level categories that effectively group one or more of the original clusters.  The goal is to find unifying themes that connect the clusters.

**Steps:**

1. **Analyze and Group:**  Carefully review the cluster descriptions in `<cluster_list>`. Identify recurring themes, patterns, or relationships. Group related clusters based on these similarities.  Use the `Scratchpad` section to detail your reasoning for each grouping.

2. **Name Generation:** For each group of clusters, generate a concise, descriptive, and unique name that captures the essence of the shared theme. The name should be informative but not overly specific.

**Guidelines for Higher-Level Cluster Names:**

*   **Thematic Consistency:** The name should clearly reflect the shared theme connecting the grouped clusters.
*   **Specificity and Clarity:** Avoid overly generic terms. The names should be specific enough to provide insight into the nature of the grouped clusters while remaining broad enough to encompass multiple clusters.
*   **Distinctiveness:** Ensure each higher-level cluster name is unique.
*   **Relevance:** Names should be relevant to the overall purpose of organizing the data (e.g., improving safety, monitoring, and observability).
*   **Neutrality:** If clusters involve sensitive topics, use professional and neutral language.

**Constraints:**

*   Generate between {low_bound_desired_number} and {upper_bound_desired_number} higher-level cluster names, aiming for {desired_number}. Adjust the number if needed to accurately reflect the cluster relationships.

**Input:**

<cluster_list>
{cluster_list}
</cluster_list>

**Scratchpad:**

[Analyze the clusters, identify common themes, and detail your reasoning for grouping specific clusters together. For each proposed group, briefly explain why those clusters were combined. Limit this section to one or two paragraphs per grouping.]

**Output Format:**

For each higher-level cluster, provide the following:

1.  **[Higher-Level Cluster Name]**
    *   **Reasoning:** [Briefly explain why these specific clusters were grouped together under this name. Highlight the key shared characteristics or themes.]
    *   **Associated Clusters:** [List the *names* of the associated clusters from the `<cluster_list>`. E.g., Project Management and Technical Collaboration Discussions, Troubleshooting Cloud and Container Management Issues]
    
"""


template_merge_clusters_examples = """
You are an expert in data analysis and categorization. You are tasked with identifying overarching themes and creating concise, descriptive higher-level names for a set of data clusters. Your goal is to group related clusters under broader categories that accurately reflect their shared characteristics, topics, and purposes. *The key to this task is to carefully analyze the EXAMPLE data points within each cluster to understand the nuances and specific topics represented.* Avoid labels that are *too narrow* (specific to a single example) or *too broad* (lacking informative detail); instead, aim for names that are informative, *appropriately* specific, and clearly differentiate the higher-level categories.

**Steps to Follow:**

1. **In-Depth Cluster Analysis:** For each cluster, meticulously examine the cluster description *AND* the provided example data points. Identify the core topics, and objectives represented. What are the concrete topics being discussed or addressed in the examples? How do the examples illustrate the cluster's described focus? Look for common nouns and subject matter within the examples. *Consider the level of abstraction: can you identify a more general subject that encompasses these specific topics?*

2. **Theme Extraction and Grouping:** Based on your analysis of the examples, identify shared themes and patterns across multiple clusters. Group clusters together that exhibit similar topics, objectives, or purposes. Justify each grouping based on specific evidence from the example data points. Consider the *purpose* behind the topics in the clusters. What overarching goal are these clusters contributing to? *Aim to identify themes that connect multiple topics across multiple clusters, avoiding themes that are too specific to a single cluster.*

3. **Higher-Level Name Generation:** Create concise, descriptive names for each higher-level cluster group. These names should capture the essence of the grouped clusters, reflecting the common topics and objectives identified in the example data. The names should be specific enough to be informative but broad enough to encompass all the associated clusters *without being so broad as to be meaningless.* Prioritize names that clearly convey the core subject matter. Avoid overly broad or vague terms. *Strive for a level of generality that captures the core essence of the grouped topics without getting bogged down in minute details.*

**Guidelines for Higher-Level Cluster Names:**

1. **Data-Driven Thematic Consistency:** Each higher-level cluster name must represent a clear and cohesive theme directly supported by the *example data points* within the encompassed clusters. A good name will be easily justifiable by referencing specific examples *while also reflecting a broader pattern of subject matter*.

2. **Appropriate Specificity and Informative Clarity:** Avoid terms that are *too narrow or too broad*. The name should provide insight into the *types* of topics without being overly restrictive. For example, "Cloud Infrastructure Management" is better than either "Troubleshooting AWS S3 Command Failures" (too specific) or "Technical Issues" (too generic).

3. **Distinctiveness and Non-Overlap:** Ensure each higher-level cluster name is unique and doesn't overlap conceptually with others. Each higher-level category should represent a distinct area of subject matter. *Consider whether two seemingly distinct categories could be merged under a more general heading.*

4. **Subject-Focused (Preferred):** Prioritize names that clearly convey the core topic matter.

5. **Contextual Relevance:** Consider the broader context of the data (e.g., a large tech company) and ensure the names are appropriate for that context.

6. **Sensitivity and Professionalism:** If any clusters touch on sensitive topics, ensure the higher-level names are professional, neutral, and avoid any potential for misinterpretation.

**Input Data:**

<cluster_list>
{cluster_list}
</cluster_list>

**Output Requirements:**

Provide at least {low_bound_desired_number} and at most {upper_bound_desired_number} higher-level cluster names, with a target of {desired_number} names. You can generate more or fewer than {desired_number} names if you feel that more or fewer are appropriate and accurately capture the clusters.

For each higher-level cluster name:

1. List the associated clusters.
2. Provide a concise justification for the grouping, explain how the examples demonstrate the shared theme. *Also, explain why the chosen name is appropriately general, capturing the essence of the grouped topics without being overly specific or too broad.*

**Output Format:**

1. **[Higher-Level Cluster Name]**
    * **Associated Clusters:** [List of cluster names]
    * **Justification (Based on Example Data and Generality):** [Detailed explanation referencing specific examples from the associated clusters. For example: "This grouping includes Clusters X and Y because both feature examples related to troubleshooting authentication errors (e.g., Cluster X's example of 'Resolving a problem with finding the certificate file for SSL' and Cluster Y's example of 'Leveraging the 'scope' claim from a JWT token...'). This indicates a shared focus on the subject of authentication. The name 'Authentication Management and Troubleshooting' is appropriately general because it encompasses various aspects of authentication, including certificate issues and token validation, without being limited to a single specific error or technology."]

2. **[Next Higher-Level Cluster Name]**
    * **Associated Clusters:** [List of cluster names]
    * **Justification (Based on Example Data and Generality):** [Detailed explanation...]

...

"""


higher_level_clusters_non_overlapping= """
You are an expert in data analysis and categorization. You are tasked with identifying overarching themes and creating concise, descriptive higher-level names for a set of data clusters. Your goal is to group related clusters under broader categories that accurately reflect their shared characteristics, subjects, and purposes. *The key to this task is to carefully analyze the EXAMPLE data points within each cluster to understand the nuances and specific subjects represented.* Avoid labels that are *too narrow* (specific to a single example) or *too broad* (lacking informative detail); instead, aim for names that are informative, *appropriately* specific, and clearly differentiate the higher-level categories. *Your PRIMARY objective is to create a complete and non-overlapping categorization, ensuring that EVERY input cluster is assigned to EXACTLY ONE higher-level cluster.*

**Steps to Follow:**

1. **In-Depth Cluster Analysis (Comprehensive):** For *every* cluster, meticulously examine the cluster description *AND* the provided example data points. Identify the core subjects, topics, and objectives represented. What are the concrete subjects being discussed or addressed in the examples? How do the examples illustrate the cluster's described focus? Look for common nouns and subject matter within the examples. *Consider the level of abstraction: can you identify a more general subject that encompasses these specific topics?* *This analysis MUST be performed for ALL clusters before proceeding to the next step.*

2. **Theme Extraction and Grouping (Mutually Exclusive):** Based on your analysis of the examples, identify shared themes and patterns across multiple clusters. Group clusters together that exhibit similar subjects, objectives, or purposes. Justify each grouping based on specific evidence from the example data points. Consider the *purpose* behind the topics in the clusters. What overarching goal are these clusters contributing to? *Aim to identify themes that connect multiple subjects across multiple clusters, avoiding themes that are too specific to a single cluster.* *Ensure that each cluster is assigned to ONLY ONE higher-level group. Do not allow any cluster to belong to multiple higher-level categories.*

3. **Higher-Level Name Generation:** Create concise, descriptive names for each higher-level cluster group. These names should capture the essence of the grouped clusters, reflecting the common subjects and objectives identified in the example data. The names should be specific enough to be informative but broad enough to encompass all the associated clusters *without being so broad as to be meaningless.* Prioritize names that clearly convey the core subject matter. Avoid overly broad or vague terms. *Strive for a level of generality that captures the core essence of the grouped topics without getting bogged down in minute details.*

**Guidelines for Higher-Level Cluster Names:**

1. **Data-Driven Thematic Consistency:** Each higher-level cluster name must represent a clear and cohesive theme directly supported by the *example data points* within the encompassed clusters. A good name will be easily justifiable by referencing specific examples *while also reflecting a broader pattern of subject matter*.

2. **Appropriate Specificity and Informative Clarity:** Avoid terms that are *too narrow or too broad*. The name should provide insight into the *types* of topics without being overly restrictive. For example, "Cloud Infrastructure Management" is better than either "Troubleshooting AWS S3 Command Failures" (too specific) or "Technical Issues" (too generic).

3. **Distinctiveness and Non-Overlap (Enforced):** Ensure each higher-level cluster name is unique and doesn't overlap conceptually with others. Each higher-level category should represent a distinct area of subject matter. *Consider whether two seemingly distinct categories could be merged under a more general heading. If a cluster seems to fit into multiple categories, prioritize the category that BEST reflects its core subject matter.*

4. **Subject-Focused (Preferred):** Prioritize names that clearly convey the core subject matter.

5. **Contextual Relevance:** Consider the broader context of the data (e.g., a large tech company) and ensure the names are appropriate for that context.

6. **Sensitivity and Professionalism:** If any clusters touch on sensitive topics, ensure the higher-level names are professional, neutral, and avoid any potential for misinterpretation.

**Input Data:**

<cluster_list>
{cluster_list}
</cluster_list>

**Output Requirements:**

Provide at least {low_bound_desired_number} and at most {upper_bound_desired_number} higher-level cluster names, with a target of {desired_number} names. You can generate more or fewer than {desired_number} names if you feel that more or fewer are appropriate and accurately capture the clusters. *It is ESSENTIAL that EVERY cluster is assigned to exactly ONE higher-level cluster. Do not leave any clusters unassigned, and do not assign any cluster to multiple higher-level clusters.*

For each higher-level cluster name:

1. List the associated clusters.
2. Provide a concise justification for the grouping, *explicitly referencing specific example data points from the associated clusters that support the grouping and the chosen name*. Explain how the examples demonstrate the shared theme. *Also, explain why the chosen name is appropriately general, capturing the essence of the grouped topics without being overly specific or too broad.* *Finally, state explicitly that this cluster grouping does not overlap with any other higher-level cluster grouping.*

**Output Format:**

1. **[Higher-Level Cluster Name]**
    * **Associated Clusters:** [List of cluster numbers and *names*]
    * **Justification (Based on Example Data and Generality):** [Detailed explanation referencing specific examples from the associated clusters. For example: "This grouping includes Clusters X and Y because both feature examples related to troubleshooting authentication errors (e.g., Cluster X's example of 'Resolving a problem with finding the certificate file for SSL' and Cluster Y's example of 'Leveraging the 'scope' claim from a JWT token...'). This indicates a shared focus on the subject of authentication. The name 'Authentication Management and Troubleshooting' is appropriately general because it encompasses various aspects of authentication, including certificate issues and token validation, without being limited to a single specific error or technology. This grouping is non-overlapping; no cluster in this group appears in any other higher-level cluster."]

2. **[Next Higher-Level Cluster Name]**
    * **Associated Clusters:** [List of cluster numbers and *names*]
    * **Justification (Based on Example Data and Generality):** [Detailed explanation...]

...

"""