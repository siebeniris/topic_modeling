naming_cluster_prompt = """
Given the following cluster of statements and a set of contrastive statements from nearby clusters, analyze both to identify the core theme of the main cluster. Carefully distinguish the subtle differences between the two sets of statements to ensure the summary and name accurately reflect the unique characteristics of the primary cluster.

Follow these steps:

1. **Analyze the Cluster Statements:** Identify the recurring topics, actions, or themes present in the primary cluster statements. Pay attention to keywords, verbs, and the overall sentiment expressed.
2. **Analyze the Contrastive Statements:** Identify the topics, actions, or themes present in the contrastive statements. Note how these differ from the primary cluster statements. Focus on what is *not* present in the primary cluster.
3. **Summarize the Primary Cluster:** Create a concise two-sentence summary that captures the essence of the primary cluster in the present tense. The summary should be specific to the cluster and clearly distinguish it from the contrastive statements. It should highlight the *unique* aspects of the primary cluster.
4. **Generate a Cluster Name:** Create a short name (at most 10 words) that reflects the overarching theme of the primary cluster statements. The name should be specific, actionable, and distinguishable from the contrastive examples. Be descriptive and assume neither good nor bad faith, even for sensitive or harmful topics. The name should be general enough to encompass the variety of statements within the primary cluster, but specific enough to differentiate it from other clusters.

Present your output in the following format:

<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>

Below are the related statements in the cluster:

<statements>
{statements}
</statements>

For context, here are statements from a nearby cluster that are NOT part of the group you're summarizing:

<contrastive_statements>
{contrastive_statements}
</contrastive_statements>

---

**Example 1:**

**Cluster Statements:**
- Discussing eSIM recognition difficulties.
- Inquiry about eSIM reissuance after rejection.
- Confirming eSIM application and KYC.

**Contrastive Statements:**
- Responding to kind words.
- Crafting a motivation statement for HR.
- Discussing meeting etiquette.

**Output:**
<summary> This cluster focuses on eSIM-related issues and inquiries, primarily concerning activation, reissuance, and application processes. It addresses troubleshooting and procedural questions related to eSIM management. </summary>
<name> eSIM Activation and Reissuance Issues </name>

---

**Example 2:**

**Cluster Statements:**
- "How to bake a chocolate cake step by step."
- "Explain the process of making sourdough bread."
- "Provide tips for decorating cupcakes."

**Contrastive Statements:**
- "Write a review of popular baking equipment."
- "Discuss the history of baking in different cultures."
- "Generate a list of famous pastry chefs."

**Output:**
<summary> This cluster focuses on step-by-step instructions and tips for baking, including cakes, bread, and cupcakes. It emphasizes practical guidance for home bakers to improve their skills. </summary>
<name> Baking instructions and tips for home bakers </name>

---

**Now, apply the same reasoning and output format to the following cluster:**

<statements>
{statements}
</statements>

<contrastive_statements>
{contrastive_statements}
</contrastive_statements>
"""