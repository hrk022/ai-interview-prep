<div align="center">

<img src="https://drive.google.com/uc?export=view&id=1cI-I2yxrHmuaxv86rLS40VNruaPaCIcW" alt="AI Interview Prep Banner" width="100%"/>

# 🎧 **AI • ML • Cybersecurity Interview Prep Simulator**  
*Your personal AI-powered interviewer, ready 24/7 to sharpen your skills.*

[![Streamlit App](https://img.shields.io/badge/Try%20It-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://ai-interview-prep-qjnexslbuefy8qgbhhgmcz.streamlit.app/)  
[![Made with LangChain](https://img.shields.io/badge/Made%20with-LangChain-blue?style=for-the-badge&logo=python)](https://www.langchain.com/)  
[![HuggingFace Datasets](https://img.shields.io/badge/Data-HuggingFace-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets)  
[![OpenAI](https://img.shields.io/badge/API-OpenAI-purple?style=for-the-badge&logo=openai)](https://openai.com/)  

</div>

---

## 🚀 **What is this?**
This isn’t your average Q&A bot.  
This is your **battlefield**, your **dojo**, your **training simulator** for crushing interviews in:
- 🤖 **Artificial Intelligence**
- 📊 **Machine Learning**
- 🔐 **Cybersecurity**

It **asks questions**, **grades your answers**, **gives feedback**, and even **motivates you like a coach who refuses to see you fail**.

---

## 🎯 **Why You’ll Love It**
- **Real interview datasets** from HuggingFace.
- **Randomized questions** every session — no memorization hacks.
- **Streaming AI responses** that feel *alive*.
- **Instant grading** with correctness, feedback, and encouragement.
- **Beautiful, minimal Streamlit UI** with side-panel controls.

---

## 🖥 **Live Demo**
⚡ **Click below & start your interview training right now:**

🎯 **[Launch the AI Interview Prep Tool](https://ai-interview-prep-qjnexslbuefy8qgbhhgmcz.streamlit.app/)**

---

## 🧠 **How It Works**
1. **Choose your domain**: AI, ML, or Cybersecurity.
2. **Introduce yourself** — the AI plays along like a real interviewer.
3. **Answer questions** — up to 10 per session.
4. **Receive instant grading**:
   - ✅ Correctness
   - 📝 Feedback
   - 💪 Encouragement
5. **Finish strong** — your AI interviewer sends you off with motivation.

---

## ⚙ **Tech Stack**
| Component | Purpose |
|-----------|---------|
| **Streamlit** | Web app & UI |
| **LangChain** | LLM orchestration |
| **HuggingFace Datasets** | Interview question bank |
| **FAISS** | Vector storage for fast retrieval |
| **Sentence-Transformers** | Embeddings for semantic search |
| **OpenAI API (Groq endpoint)** | LLM backend |
| **Python-dotenv** | Environment variable handling |

---

## 🛠 **Installation**
```bash
# 1️⃣ Clone this repo
git clone https://github.com/hrk022/ai-interview-prep.git
cd ai-interview-prep

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add your API key
echo "OPENAI_API_KEY=your_api_key_here" > open_ai.env

# 4️⃣ Run it locally
streamlit run main.py
