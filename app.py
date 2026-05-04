import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# local embedding model, to avoid API issues
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Load ENV ─────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "bootcamp-guide-chatbot.pdf")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
RETRIEVER_K = 2
MAX_CONTEXT_CHARS = 2500

# ── 1. Config Page ───────────────────────────────────
st.set_page_config(page_title="Gemini Chatbot", layout="centered")

st.title("Data Sceince Guide Chatbot 🤖")
st.caption("Simple chatbot for learning Data Science as bootcamp student, with Google Gemini API")

# ── 2. Sidebar ───────────────────────────────────────
with st.sidebar:
    st.subheader("Settings")

    manual_key = st.text_input("Override API Key (optional)", type="password")

    reset_button = st.button("Reset Chat")

# prioritize with manual key inputted by user
google_api_key = manual_key if manual_key else API_KEY

# ── 3. Validasi API Key ─────────────────────────────
if not google_api_key:
    st.warning("Missing API Key, please input API key first.")
    st.stop()

# ── 4. Init Client ───────────────────────────────────
# create the new client only if the client not created yet OR user change API key on the sidebar
if ("genai_client" not in st.session_state) or (
    getattr(st.session_state, "_last_key", None) != google_api_key
):
    try:
        # create new Gemini client with API key from the sidebar
        st.session_state.genai_client = genai.Client(api_key=google_api_key)

        # save the last inputted key, and detect if there are new key (if any)
        st.session_state._last_key = google_api_key

        # if the key changes, delete last chat session (chat and messages)
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)

    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

# ── 5. Init Vectorstore ────────────────────────────
# only create vector database if not created yet
if "vectorstore" not in st.session_state:
    try:
        # create embeddings with HuggingFaceEmbeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # check whether the FAISS index is initialized or not
        if os.path.exists(FAISS_INDEX_PATH):
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # load RAG pdf
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()

            # extract text from pdf and chunk the document
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # set chunk size 1000
                chunk_overlap=200 # and 200 chunk for overlap
            )
            docs = splitter.split_documents(documents)

            # initialize FAISS and load documents
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            # save FAISS to the local disk, 
            # so not embedd from scratch if the program restarted
            vectorstore.save_local(FAISS_INDEX_PATH)
        
        # save the vectorstore as vector database
        st.session_state.vectorstore = vectorstore

    except Exception as e:
        st.error(f"Vectorstore error: {e}")
        st.stop()

# ── 6. Init Chat ─────────────────────────────────────
# initialize the Gemini session chat if not existed yet
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.genai_client.chats.create(
        model="gemini-2.5-flash" # use the model
    )

# initialize messages history if not existed yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 7. Reset ─────────────────────────────────────────
# if the reset button clicked, clear the session
if reset_button:
    st.session_state.clear() # also clear the chat and messages
    st.rerun()

# ── 8. Show Chat ─────────────────────────────────────
# looping to show all messages in session_state
# if this script is being re-run, all of the messages will be shown
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): # show the bubble chat with respective role
        st.markdown(msg["content"])

# ── 9. Input ─────────────────────────────────────────
prompt = st.chat_input("Ask anything about Data Science...")

if prompt: # only run below, if user send a chat (a prompt)
    # 1. add the message to the history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 2. show the bubble of user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. retrieve the context from vector database
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    retrieved_docs = retriever.invoke(prompt)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])[:MAX_CONTEXT_CHARS]

    # 4. augmented the prompt
    augmented_prompt = f"""
        Kamu adalah **Bootcamp Guide Assistant** yang membantu student memahami 
        Data Analyst, Data Scientist, dan Data Engineer secara praktis.

        🎯 TUJUAN UTAMA:
        - Membantu user belajar konsep data (DA, DS, DE) dengan cara yang mudah dipahami
        - Fokus ke practical understanding + intuition (bukan teori berat)
        - Memberikan penjelasan yang relevan dengan konteks bootcamp

        🧠 ATURAN MENJAWAB:
        1. Gunakan **bahasa Indonesia yang natural dan mudah dipahami**
        2. Jika user bertanya dalam bahasa Inggris, tetap jawab dalam bahasa Indonesia
        3. Jawaban harus:
        - jelas
        - singkat dan langsung menjawab inti pertanyaan
        - 1 paragraf pendek untuk pertanyaan sederhana
        - 2 paragraf pendek hanya jika perlu penjelasan tambahan
        - rencanakan jawaban agar selesai utuh; jangan berhenti di tengah kalimat
        - gunakan bullet point hanya kalau sangat membantu, maksimal 3 bullet pendek

        📚 PENGGUNAAN CONTEXT:
        - Gunakan konteks di bawah sebagai referensi utama
        - Jika informasi relevan ditemukan → gunakan untuk menjawab
        - Jika konteks hanya sebagian relevan → boleh tambahkan penjelasan sederhana 
        (tanpa mengarang berlebihan)
        - Jika tidak ada hubungannya sama sekali → arahkan user dengan sopan

        🚫 BATASAN DOMAIN:
        Jika pertanyaan TIDAK berhubungan dengan:
        - Data Analyst
        - Data Scientist
        - Data Engineer
        - Data Analytics / Business Analytics

        Maka:
        - Jangan jawab langsung
        - Arahkan dengan sopan bahwa chatbot ini fokus ke bidang data
        - Contoh arahkan: "Aku fokus bantu di bidang data seperti Data Analyst, 
        Data Scientist, atau Data Engineer..."

        💡 GAYA JAWABAN:
        - Edukatif (seperti mentor bootcamp)
        - Tidak terlalu formal
        - Fokus ke "biar ngerti", bukan "biar keliatan pintar"
        - Jangan memberi penutup panjang, checklist besar, atau daftar tambahan kecuali diminta user
        - Lebih baik jawaban pendek tapi selesai, daripada banyak poin tapi terpotong

        ━━━━━━━━━━━━━━━━━━━
        📄 CONTEXT:
        {context}

        ❓ PERTANYAAN:
        {prompt}
        """

    # 5. send the prompt and context to the Gemini and show response
    try:
        # send the chat to Gemini with current chat session
        response = st.session_state.chat.send_message(augmented_prompt)

        # get the answer text from response and check the text attribute
        # to prevent error if the response format is not text
        answer = response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        # if there is error (e.g.: resource exhausted, or connection issues), show the error
        answer = f"Error: {e}"

    # 6. show the bubble response of assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    # 7. save the assistant response to the messages history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
