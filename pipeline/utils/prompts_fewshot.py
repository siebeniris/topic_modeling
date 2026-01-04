convo_prompt = """
Analyze the provided conversation turn between a human user and the AI assistant. The conversation turn to be analyzed is specified in <Conversation to Analyze>, while <System Message> and <Previous Conversation> provide additional context if available. Use the examples and rules provided to ensure accurate and consistent analysis. Your analysis should include the following components:

<Instructions>
Task: Identify the user's overall task for the assistant. Use active wording to describe what the user is asking the assistant to do (e.g., translating a message, answering a question, debugging a program, writing an email). For translation, specify both the source and target languages.

Subject: Determine the subject or topic of the content being discussed. Use active wording to describe the focus of the conversation (e.g., billing question, translation request). For translation, focus on what is being translated.

Task Keyword: Provide a short phrase summarizing the action.

Subject Keyword:  Provide a short phrase summarizing the topic.

Language: Identify the main languages and the language script type used in the conversation. If multiple languages are used, list all of them (e.g., "Japanese, Kanji script; Vietnamese, Latin script").

Translation Direction: If translation is involved, specify the direction (e.g., "Japanese to Vietnamese"); otherwise, "NA".

Concerning: On a scale of 1 to 5, assess how concerning or potentially harmful the conversation is from a safety perspective. Use the rubric provided below.

Summary: Provide a single coherent sentence summarizing the task and subject, focusing on the user's request and the topic of the conversation.

Relevance: 
- If the system message is missing, empty, or the string "None", answer "NA".
- If the system message is present, answer "yes" ONLY if the conversation DIRECTLY and CLEARLY aligns with the system message's purpose. If the conversation deviates from the system message's intended topic or function, answer "no".
- State your reasoning before giving your answer.

Conversation Continuation:
- If the previous conversation turn is missing, empty, or the string "None", answer "NA".
- If the previous conversation turn is present, answer "yes" ONLY if the current conversation is a DIRECT continuation of the previous turn, addressing the same topic or task. If the conversation shifts to a new topic or task, answer "no".
- State your reasoning before giving your answer.


<Rules>
Do not include any personally identifiable information (PII), such as names, locations, phone numbers, or email addresses.
Do not include any proper nouns.
Limit your response to at most three sentences per question, except for the Summary, which must be one sentence only.
When assessing the "Concerning" level, base your judgment on the user's request and the topic of the conversation.

<Rubric for Concerning Flag>
[1]: Not concerning at all. Completely benign conversation.
Example: A friendly chat about favorite books or movies.
[2]: Slightly concerning, but likely harmless.
Example: A heated but civil debate about politics.
[3]: Moderately concerning. May warrant a closer look.
Example: Discussion of controversial topics that could escalate.
[4]: Very concerning. Likely needs review.
Example: Conversations involving potential self-harm or harmful ideation.
[5]: Extremely concerning. Immediate review needed.
Example: Explicit threats of violence or illegal activities.

<Output Format>
Task: [Answer here].
Subject: [Answer here].
Task Keyword: [Insert one descriptive phrase].
Subject Keyword: [Insert one descriptive phrase].
Language: [Answer here].
Translation Direction: [Answer here].
Concerning: [Answer here].
Summary: [Insert one coherent sentence summarizing the task and subject].
Relevance: [Answer here].
Conversation Continuation: [Answer here].


<Examples>

<Example 1>
System Message: You are an expert in translation.
Previous Conversation: None
Conversation Turn:
Human: 12月使用分の料金が間違ってるように見えますがどこに問い合わせればいいですか？
AI: Tôi thấy có vẻ như chi phí sử dụng trong tháng 12 bị sai. Tôi nên liên hệ ở đâu để được giải đáp?

Analysis:
Task: Translating a billing inquiry from Japanese to Vietnamese.
Subject: Asking about incorrect charges for December usage.
Task Keyword: "Translation task."
Subject Keyword: "Billing inquiry."
Language: Japanese (Kanji script), Vietnamese (Latin script).
Translation Direction: Japanese to Vietnamese.
Concerning: 1
Summary: A billing inquiry about incorrect December charges was translated from Japanese to Vietnamese.
Relevance: Yes (The system message is present and the conversation is a translation).
Conversation Continuation: NA (The previous conversation is not present).


<Example 2>
System Message: None
Previous Conversation: None
Conversation Turn:
Human: Hi, I need help drafting a reply to this email:
"Thank you for confirming your upcoming interview. If you have any questions regarding scheduling logistics, feel free to ask me. I will be available to answer any questions regarding the content of your interview and/or how to prepare."
AI: Here is an example reply:

Subject: Re: Interview Confirmation
Dear [Recipient],
Thank you for your email and for confirming the details of my upcoming interview. I appreciate your support and will reach out if I have any questions regarding the scheduling logistics.
I will also keep in touch with you for any questions related to the interview content or preparation.
Thank you again for your assistance, and I look forward to the interview.
Best regards,
[Your Name]

Analysis:
Task: Drafting a reply to an email.
Subject: Responding to an interview confirmation email.
Task Keyword: "Drafting a reply."
Subject Keyword: "Interview confirmation."
Language: English (Latin script).
Translation Direction: NA.
Concerning: 1
Summary: A reply was drafted to respond to an interview confirmation email.
Relevance: NA (The system message is not present).
Conversation Continuation: NA (The previous conversation is not present).


<Example 3>
System Message: You are a helpful assistant that can provide information about animals.
Previous Conversation: None
Conversation Turn:
Human: What is the capital of France?
AI: The capital of France is Paris.

Analysis:
Task: Answering a question.
Subject: The capital of France.
Task Keyword: "Answering question."
Subject Keyword: "Capital of France."
Language: English (Latin script).
Translation Direction: NA.
Concerning: 1
Summary: The AI answered a question about the capital of France.
Relevance: No (The system message specifies animals, but the conversation is about geography).
Conversation Continuation: NA (The previous conversation is not present).

<Example 4>
System Message: You are a helpful assistant that can provide information about animals.
Previous Conversation:
Human: Tell me about lions.
AI: Lions are large cats that live in Africa.
Conversation Turn:
Human: What is the capital of France?
AI: The capital of France is Paris.

Analysis:
Task: Answering a question.
Subject: The capital of France.
Task Keyword: "Answering question."
Subject Keyword: "Capital of France."
Language: English (Latin script).
Translation Direction: NA.
Concerning: 1
Summary: The AI answered a question about the capital of France.
Relevance: No (The system message specifies animals, but the conversation is about geography).
Conversation Continuation: No (The previous conversation was about lions, but this question is about the capital of France).


<System Message>
{message}

<Previous Conversation>
{previous_conversation}

<Conversation to Analyze>
{input}
"""