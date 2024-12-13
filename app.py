import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch


st.title("Feedback Sentiment analyzer!!!")

model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434/")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant."),
        ("human","Generate a thank you note for this positive feedback: {feedback}."),
    ]
)
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant."),
        ("human","Generate a response addressing this negative feedback: {feedback}."),
    ]
)
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant."),
        ("human","Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)
esclate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant."),
        ("human","Generate a message to esclate this feedback to a human agent: {feedback}."),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Classify the sentiment of this feedback as positive , negative , neutral or esclate:{feedback}.")
    ]
)

branches = RunnableBranch(
    (
        lambda x : "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x : "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x : "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    esclate_feedback_template | model | StrOutputParser()
)

classification_chain  = classification_template | model | StrOutputParser() #1 2

# chain = classification_chain | branches #1

with st.form("llm-form"):
    review = st.text_area("Enter your feedback here.")
    submit = st.form_submit_button("Submit")
    
if submit and review:
    with st.spinner("Generating response..."):
        emotion = classification_chain.invoke({"feedback":review}) #2
        st.write(f"Sentiment of this feedback is : {emotion}") #2
        st.write("---") #2
        result = branches.invoke(emotion) #2
        st.write(result) #2
        
        
        # result = chain.invoke({"feedback":review}) #1
        # st.write(result) #1