# Annotation Guidelines for Task Summarization and Clustering

## I. Goal 

The goal of this annotation task is to determine the accuracy and relevance of topic clusters assigned to individual conversation turns. Annotators will assess whether the provided cluster name accurately reflects the content of the conversation turn and whether the task summary aligns with both the conversation turn and the assigned cluster.

## II. Input Data :

For each data point, you will be provided with the following information:

Conversation ID: The id number for the thread of the conversation turn.

Turn ID: The id number for the turn of the conversation in a conversation thread.

Conversation Turn: The actual text of the user and AI utterance in the conversation.

Task Summary: A brief summary of the task being performed in the conversation turn.

Sub-Cluster Name: The name of the topic cluster assigned to this conversation turn. This represents the predicted topic of the turn.

Higher-Level Cluster Name: A broader, more general category that the cluster belongs to. This provides context for the cluster name.

## III. Annotation Task:

For each data point, you will perform the following two evaluations and provide a rating:

__Relevance of Sub-Cluster Name / Higher Cluster Name to Conversation Turn__: How well does the Cluster Names represent the topic discussed in the Conversation Turn?

__Consistency of Task Summary__: How well does the Task Summary align with the Conversation Turn and the Cluster Name? Does the summary accurately describe the task being performed in the turn, given the identified topic?

## IV. Annotation: Match/No Match (Binary Rating)

The goal is to determine if the provided Sub-Cluster Name, Higher Cluster Name, or Task Summary accurately and meaningfully describes the content of the Conversation Turn.

Choose ONE of the following options:

__Match__: The Sub-Cluster Name, Higher-Level Cluster Name, or Task Summary accurately reflects a significant and primary topic or purpose of the Conversation Turn. This means:

* Key topics are covered: The name/summary should capture the main subject matter discussed in the conversation turn.

* Accurate representation: The description shouldn't be misleading or misrepresent the conversation.

* Understandable without extra context: Someone reading the cluster names and summary should get a reasonable understanding of what the conversation turn is about without needing to read other parts of the conversation.

* Think: Does the sub-cluster name, higher-level cluster name, summary provide a useful and direct label for this conversation turn, respectively?

__No Match__: The Sub-Cluster Name, Higher Cluster Name, or Task Summary fails to accurately and meaningfully describe the Conversation Turn. This includes cases where:

* Tangential Connection: The sub-cluster name/higher-level cluster name/summary is only vaguely related or mentions a minor point. The connection requires significant interpretation or inference to understand.

* Misleading: The sub-cluster name/higher-level cluster name/summary is actively misleading or suggests a topic that is not the primary focus of the conversation turn.

* Incomplete: The sub-cluster name/higher-level cluster name/summary omits crucial information or key aspects of the conversation turn, making it an inadequate representation.

* Unrelated: The sub-cluster name/higher-level cluster name/summary is completely unrelated to the content of the conversation turn.

* Think: Does the sub-cluster name/higher-level cluster name/summary provide a poor, unhelpful, or incorrect label for this conversation turn?

## V. Detailed Instructions:

Read the Conversation Turn: Carefully read the text of the conversation turn to understand the user's intent and the agent's response.

Understand the Task Summary: Understand what the task summary is conveying about the conversation turn.

__Evaluate Cluster Name Relevance:__

Consider the Cluster Name and the Higher-Level Cluster Name. Do they provide a reasonable categorization for the conversation turn?Does the Cluster Name accurately reflect the main topic or subject being discussed in the Conversation Turn? Assign a match/no-match based on the guideline.

__Evaluate Task Summary Consistency:__

Does the Task Summary accurately describe what's happening in the Conversation Turn?Is the Task Summary consistent with the assigned Cluster Name? Would someone familiar with the cluster expect this type of task to be associated with it? Assign a match/no-match based on the guideline.

Record Your Annotations: Record your ratings for "Relevance of Sub-Cluster Name", “Relevance of Higher-Level Cluster Name” and "Consistency of Task Summary" for each data point.

