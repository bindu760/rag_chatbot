import os
import hashlib
import streamlit as st

from dotenv import load_dotenv

from llama_parse import LlamaParse

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# ------------------------------------
# Load .env
# ------------------------------------

load_dotenv()

LLAMA_PARSE_KEY = os.getenv("LLAMA_PARSE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------------
# Page
# ------------------------------------

st.set_page_config(
    page_title="Multi PDF RAG",
    layout="wide"
)

st.title("📄 Multi PDF Parser + RAG")

# ------------------------------------
# Check API Keys
# ------------------------------------

if not LLAMA_PARSE_KEY:
    st.error("LLAMA_PARSE_KEY not found inside .env")
    st.stop()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found inside .env")
    st.stop()

# ------------------------------------
# Sidebar
# ------------------------------------

with st.sidebar:

    st.header("Settings")

    groq_model = st.selectbox(

        "Groq Model",

        [

            "llama-3.3-70b-versatile",

            "llama-3.1-8b-instant",

            "mixtral-8x7b-32768",

            "gemma2-9b-it"

        ]

    )

    chunk_size = st.slider(

        "Chunk Size",

        128,

        2048,

        512,

        step=64

    )

    chunk_overlap = st.slider(

        "Chunk Overlap",

        0,

        256,

        50,

        step=10

    )

    top_k = st.slider(

        "Top K",

        1,

        10,

        4

    )

    parse_instruction = st.text_area(

        "Parsing Instruction",

        value="Parse this document into clean markdown while preserving headings and tables."

    )

    if st.button("Clear Everything"):

        for key in [

            "parsed_docs",

            "chunks",

            "index",

            "qa_history",

            "processed_hash"

        ]:

            st.session_state.pop(key, None)

        st.rerun()

# ------------------------------------
# Session State
# ------------------------------------

st.session_state.setdefault(

    "parsed_docs",

    {}

)

st.session_state.setdefault(

    "chunks",

    []

)

st.session_state.setdefault(

    "index",

    None

)

st.session_state.setdefault(

    "qa_history",

    []

)

st.session_state.setdefault(

    "processed_hash",

    None

)

# ------------------------------------
# Upload PDFs
# ------------------------------------

st.header("Upload PDF Files")

uploaded_files = st.file_uploader(

    "Upload one or more PDFs",

    type=["pdf"],

    accept_multiple_files=True

)

# ------------------------------------
# Signature
# ------------------------------------

def files_signature(

    files,

    chunk_size,

    chunk_overlap,

    instruction

):

    h = hashlib.sha256()

    for file in files:

        h.update(file.name.encode())

        h.update(str(file.size).encode())

    h.update(str(chunk_size).encode())

    h.update(str(chunk_overlap).encode())

    h.update(instruction.encode())

    return h.hexdigest()

# ------------------------------------
# Parse PDFs
# ------------------------------------

def parse_pdfs(files, instruction):

    parser = LlamaParse(
        api_key=LLAMA_PARSE_KEY,
        result_type="markdown",
        parsing_instruction=instruction,
        verbose=True
    )

    parsed_docs = {}

    progress = st.progress(0)

    for i, file in enumerate(files):

        temp_path = f"temp_{file.name}"

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        progress.progress((i + 1) / len(files))

        docs = parser.load_data(temp_path)

        text = "\n\n".join([doc.text for doc in docs])

        parsed_docs[file.name] = text

        os.remove(temp_path)

    progress.empty()

    return parsed_docs

# ------------------------------------
# Chunk Documents
# ------------------------------------

def chunk_docs(parsed_docs, chunk_size, chunk_overlap):

    splitter = SentenceSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap

    )

    all_nodes = []

    for filename, text in parsed_docs.items():

        document = Document(

            text=text,

            metadata={

                "source": filename

            }

        )

        nodes = splitter.get_nodes_from_documents([document])
        st.write(filename, "Chunks:", len(nodes))

        for node in nodes:

            node.metadata["source"] = filename

        all_nodes.extend(nodes)

    return all_nodes

# ------------------------------------
# Build Vector Index
# ------------------------------------

def build_index(nodes, groq_model):

    Settings.embed_model = HuggingFaceEmbedding(

        model_name="BAAI/bge-small-en-v1.5"

    )

    Settings.llm = Groq(

        model=groq_model,

        api_key=GROQ_API_KEY

    )

    index = VectorStoreIndex(nodes)

    return index

# ------------------------------------
# Automatic Pipeline
# ------------------------------------

if uploaded_files:

    signature = files_signature(

        uploaded_files,

        chunk_size,

        chunk_overlap,

        parse_instruction

    )

    if signature != st.session_state.processed_hash:

        with st.status(

            "Processing PDFs...",

            expanded=True

        ):

            st.write("Parsing PDFs...")

            st.session_state.parsed_docs = parse_pdfs(

                uploaded_files,

                parse_instruction

            )

            st.write("Creating Chunks...")

            st.session_state.chunks = chunk_docs(

                st.session_state.parsed_docs,
                chunk_size,
                chunk_overlap

            )

            st.write("Building Vector Index...")

            st.session_state.index = build_index(

                st.session_state.chunks,
                groq_model

            )

            st.session_state.processed_hash = signature

            st.session_state.qa_history = []

        st.success("Pipeline Completed Successfully!")
        st.write("Parsed PDFs:")
        st.write(st.session_state.parsed_docs.keys())

# ------------------------------------
# View Parsed Documents
# ------------------------------------

if st.session_state.parsed_docs:

    st.header("📄 Parsed PDF")

    selected_pdf = st.selectbox(

        "Select PDF",

        list(st.session_state.parsed_docs.keys())

    )

    with st.expander("View Parsed Markdown", expanded=True):

        st.markdown(

            st.session_state.parsed_docs[selected_pdf]

        )

# ------------------------------------
# View Chunks
# ------------------------------------

if st.session_state.chunks:

    st.header("📦 Chunks")

    st.write(

        f"Total Chunks : {len(st.session_state.chunks)}"

    )

    filter_pdf = st.selectbox(

        "Filter",

        ["All"] + list(st.session_state.parsed_docs.keys())

    )

    if filter_pdf == "All":

        nodes = st.session_state.chunks

    else:

        nodes = [

            n

            for n in st.session_state.chunks

            if n.metadata["source"] == filter_pdf

        ]

    for i, node in enumerate(nodes):

        with st.expander(

            f"Chunk {i+1} | {node.metadata['source']}"

        ):

            st.text(node.text)

# ------------------------------------
# Ask Questions
# ------------------------------------

if st.session_state.index:

    st.header("💬 Ask Questions")

    question = st.chat_input(

        "Ask anything about uploaded PDFs..."

    )

    if question:

        with st.chat_message("user"):

            st.markdown(question)

        retriever = st.session_state.index.as_retriever(

            similarity_top_k=top_k

        )

        query_engine = st.session_state.index.as_query_engine(

            similarity_top_k=top_k

        )

        with st.spinner("Thinking..."):

            retrieved_nodes = retriever.retrieve(question)

            response = query_engine.query(question)

        answer = str(response)

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.qa_history.insert(

            0,

            {

                "question": question,

                "answer": answer,

                "nodes": retrieved_nodes

            }

        )

# ------------------------------------
# Chat History
# ------------------------------------

if st.session_state.qa_history:

    st.header("📝 Previous Questions")

    for item in st.session_state.qa_history:

        with st.expander(item["question"]):

            st.markdown(

                f"### Answer\n\n{item['answer']}"

            )

            pdfs = sorted(

                list(

                    set(

                        node.metadata["source"]

                        for node in item["nodes"]

                    )

                )

            )

            st.success(

                "Sources : "

                + ", ".join(pdfs)

            )

            st.markdown("### Retrieved Chunks")

            for i, node in enumerate(item["nodes"]):

                st.markdown(

                    f"#### Chunk {i+1}"

                )

                st.caption(

                    f"Source : {node.metadata['source']}"

                )

                if hasattr(node, "score"):

                    st.write(

                        f"Similarity Score : {node.score:.4f}"

                    )

                st.text(

                    node.node.get_content()

                )

                st.divider()