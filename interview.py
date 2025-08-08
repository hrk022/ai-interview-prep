import os
import streamlit as st
import json
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler

# ---------------------- ENV SETUP ----------------------
load_dotenv("open_ai.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# ---------------------- STREAM HANDLER ----------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="AI Interview Bot", page_icon="🎧")
st.title("🎧 AI Interview Simulator")

chattiness = st.sidebar.slider("Chattiness Level 💬", 1, 10, 7)
max_tokens_slider = st.sidebar.slider("Maximum Tokens 📌", 20, 800, 300)
temperature = 0.3 + (chattiness - 1) * 0.08
max_tokens = max_tokens_slider
selected_domain = st.sidebar.radio("Choose your domain:", ["AI", "ML", "Cybersecurity"])


QUESTIONS_PER_SESSION = 10

# ---------------------- DOMAIN DATA HANDLER ----------------------
@st.cache_data
def load_documents(domain):
    if domain == "AI":
        dataset = load_dataset("K-areem/AI-Interview-Questions")["train"]
    elif domain == "ML":
        dataset = load_dataset("manasuma/ml_interview_qa")["train"]
    else:
        dataset = load_dataset("pAILabs/infosec-security-qa")["train"]

    documents = []
    for item in dataset:
        for val in item.values():
            if isinstance(val, str):
                val = val.strip().replace("[INST]", "").replace("[/INST]", "").replace("<s>", "").replace("</s>", "")
                if len(val) > 10 and not val.startswith("Answer:"):
                    documents.append(Document(page_content=val))
    return documents

def create_retriever_from_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    split_docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}), split_docs

def get_random_questions(documents, n):
    import random
    random.shuffle(documents)
    return [doc.page_content for doc in documents[:n]]

# ---------------------- SESSION STATE INIT ----------------------
default_keys = {
    "current_domain": selected_domain,
    "intro_done": False,
    "question_number": 1,
    "questions": [],
    "current_question": "",
    "completed": False,
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    "qa_chain": None,
    "grading_chain": None
}
for key, val in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------- DOMAIN SWITCH HANDLER ----------------------
def switch_domain_if_needed():
    if selected_domain != st.session_state.current_domain or not st.session_state.questions:
        st.session_state.current_domain = selected_domain
        st.session_state.intro_done = False
        st.session_state.question_number = 1
        st.session_state.completed = False
        st.session_state.memory.clear()

        # Load and build retriever
        documents = load_documents(selected_domain)
        retriever, all_docs = create_retriever_from_documents(documents)

        # Random question set
        st.session_state.questions = get_random_questions(all_docs, QUESTIONS_PER_SESSION)
        st.session_state.current_question = ""

        # QA chain
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Provide a clear and complete answer.
"""
        )
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                model_name="llama3-70b-8192",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                streaming=True,
            ),
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        # Grading chain
        grading_prompt = PromptTemplate(
            input_variables=["question", "answer", "ideal_answer"],
            template="""
You are an evaluator grading a user's interview answer.

Question: {question}
User's Answer: {answer}
Ideal Answer: {ideal_answer}
- keep the Ideal Answer very concise

Evaluate the correctness of the answer. Respond with one of the following:
- "Correct"
- "Incorrect"
Also provide:
- Feedback: A 1-2 line explanation.
- Encouragement: A 1-line motivational suggestion.

Return in this JSON format:
{{"Correctness": "...", "Feedback": "...", "Encouragement": "..."}}
"""
        )
        st.session_state.grading_chain = LLMChain(
            llm=ChatOpenAI(
                model_name="llama3-70b-8192",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                streaming=False,
            ),
            prompt=grading_prompt
        )

# ---------------------- INITIALIZE ----------------------
switch_domain_if_needed()

# ---------------------- INTRO ----------------------
if not st.session_state.intro_done:
    with st.chat_message("assistant"):
        st.markdown(f"👋 Welcome to your **{selected_domain}** interview practice!")
        st.markdown("Please start with a short introduction about yourself.")

# ---------------------- USER INPUT ----------------------
user_input = st.chat_input("Your response...")

if user_input:
    st.chat_message("user").markdown(user_input)

    if not st.session_state.intro_done:
        with st.chat_message("assistant"):
            st.markdown("✅ Thanks for the introduction! Let's begin with the questions.")
        st.session_state.intro_done = True
        st.session_state.current_question = st.session_state.questions[0]

    elif not st.session_state.completed:
        question = st.session_state.current_question

        with st.chat_message("assistant"):
            response_container = st.empty()
            stream = StreamHandler(response_container)
            response = st.session_state.qa_chain.invoke(
                {"question": question}, config={"callbacks": [stream]}
            )
            ideal_answer = response.get("answer", "")

        grading_output = st.session_state.grading_chain.invoke({
            "question": question,
            "answer": user_input,
            "ideal_answer": ideal_answer
        })

        try:
            result = json.loads(grading_output["text"])
            with st.chat_message("assistant"):
                st.markdown(f"**Correctness:** {result['Correctness']}")
                st.markdown(f"**Feedback:** {result['Feedback']}")
                st.markdown(f"**Encouragement:** {result['Encouragement']}")
        except Exception as e:
            st.error("Evaluation failed. Try again or refine the answer.")

        # Move to next question
        st.session_state.question_number += 1
        if st.session_state.question_number > QUESTIONS_PER_SESSION:
            st.session_state.completed = True
        else:
            next_index = st.session_state.question_number - 1
            if next_index < len(st.session_state.questions):
                st.session_state.current_question = st.session_state.questions[next_index]
            else:
                st.session_state.completed = True

# ---------------------- NEXT QUESTION ----------------------
if st.session_state.intro_done and not st.session_state.completed:
    st.chat_message("assistant").markdown(
        f"**Question {st.session_state.question_number}/{QUESTIONS_PER_SESSION}**:\n{st.session_state.current_question}"
    )

# ---------------------- FINAL MESSAGE ----------------------
if st.session_state.completed:
    with st.chat_message("assistant"):
        st.success("🎉 Interview Complete! You’ve done a great job. Keep practicing and good luck with your real interviews!")
