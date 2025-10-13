import os
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import PyPDF2
from langchain.schema import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

# Load environment variables
load_dotenv()

# ================================
# CONFIGURE YOUR DEFAULT DOCUMENT PATH HERE
# ================================
DEFAULT_DOCUMENT_PATH = ""  # Replace with your actual file path
# Supported formats: PDF, DOC, DOCX

# Page configuration
st.set_page_config(
    page_title="Question Generator", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

def load_default_document():
    """Load the default document"""
    if not os.path.exists(DEFAULT_DOCUMENT_PATH):
        raise FileNotFoundError(f"Document not found: {DEFAULT_DOCUMENT_PATH}")
    
    file_extension = DEFAULT_DOCUMENT_PATH.split('.')[-1].lower()
    docs = []
    
    if file_extension == "pdf":
        with open(DEFAULT_DOCUMENT_PATH, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": os.path.basename(DEFAULT_DOCUMENT_PATH), "page": page_num + 1}
                    )
                    docs.append(doc)
    
    elif file_extension in ["doc", "docx"]:
        loader = Docx2txtLoader(DEFAULT_DOCUMENT_PATH)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(DEFAULT_DOCUMENT_PATH)
    
    else:
        raise ValueError(f"Unsupported format: {file_extension}")
    
    return docs

def create_question_prompt():
    """Create specialized prompt for question generation"""
    import random
    
    command_terms = ["Evaluate", "Discuss", "Evaluate"] 
    random.shuffle(command_terms)
    
    prompt = f"""
You are an expert examiner creating examination questions based on the provided document content.

STRICT REQUIREMENTS:
1. Generate exactly 3 questions, each with parts (a) and (b)
2. Part (a) must be worth 10 marks and use "Explain" command term
3. Part (b) must be worth 15 marks - Question 1 use "{command_terms[0]}", Question 2 use "{command_terms[1]}", Question 3 use "{command_terms[2]}"
4. Questions must be directly related to the document content
5. Include real-world examples requirement in part (b)

COMMAND TERM RULES FROM YOUR SPREADSHEET:
- EVALUATE: To be used to weigh up the strengths and limitations of an economic policy or concept. USE OF SUPERLATIVES: Applicable in some cases - not all. SUPERLATIVE EXAMPLES: Most significant. TYPE: 15 markers. FREQUENCY: High
- EXPLAIN: Give a detailed account including reasons or causes. USE OF SUPERLATIVES: No. SUPERLATIVE EXAMPLES: Not applicable. TYPE: 10 markers. FREQUENCY: High  
- DISCUSS: To be used to give a considered and balanced review that includes a range of arguments, factors, or hypotheses. USE OF SUPERLATIVES: Yes - in 90% of the questions. SUPERLATIVE EXAMPLES: Most significant. TYPE: 15 markers. FREQUENCY: High

SUPERLATIVES IMPLEMENTATION:
- EVALUATE questions: Use superlatives only when appropriate (some cases)
- DISCUSS questions: Must use superlatives in 90% of questions
- Choose superlatives that naturally fit the document content (most/least significant, most/least effective, best/worst approach, always/never, etc.)

QUESTION FORMAT VARIATIONS:
- Vary the question starters: "Explain how...", "Explain why...", "Explain the relationship between...", "Explain the process of...", "Explain the impact of..."
- For part (b): Mix between "Using real-world examples, evaluate..." and "Using real-world examples, discuss..." and "With reference to real-world examples, evaluate..." etc.

Document Content: {{context}}

Generate 3 examination questions now with varied formats:
"""
    return prompt

def create_pdf(questions_text):
    """Create PDF from questions"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    story = []
    story.append(Paragraph("Examination Questions", title_style))
    story.append(Spacer(1, 20))
    
    lines = questions_text.split('\n')
    for line in lines:
        if line.strip():
            line = line.replace('**', '<b>').replace('**', '</b>')
            if line.startswith('##'):
                line = line.replace('##', '').strip()
                para = Paragraph(f"<b>{line}</b>", styles['Heading2'])
            elif line.startswith('#'):
                line = line.replace('#', '').strip()
                para = Paragraph(f"<b>{line}</b>", styles['Heading2'])
            else:
                para = Paragraph(line, normal_style)
            story.append(para)
            story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_download_link(pdf_buffer, filename):
    """Generate download link"""
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background-color: #28a745; color: white; padding: 0.5rem 1rem; border: none; border-radius: 5px; cursor: pointer;">Download PDF</button></a>'

# Header
st.markdown('<h1 class="main-header">Question Generator</h1>', unsafe_allow_html=True)

# Initialize document processing
if not api_key:
    st.error("OpenAI API Key required in .env file")
    st.stop()

# Auto-load document
if not st.session_state.get("document_processed", False):
    if not os.path.exists(DEFAULT_DOCUMENT_PATH):
        st.error("Document not found. Please check the file path in the code.")
        st.stop()
    
    with st.spinner("Preparing document..."):
        try:
            docs = load_default_document()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            texts = []
            for doc in docs:
                chunks = splitter.split_text(doc.page_content)
                texts.extend(chunks)
            
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectordb = FAISS.from_texts(
                texts, 
                embeddings, 
                metadatas=[{"source": os.path.basename(DEFAULT_DOCUMENT_PATH)}] * len(texts)
            )
            
            st.session_state.vectordb = vectordb
            st.session_state.document_processed = True
            st.session_state.document_name = os.path.basename(DEFAULT_DOCUMENT_PATH)
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.stop()

# Document ready indicator (minimal)
if st.session_state.get("document_processed", False):
    st.success(f"📄 {st.session_state.document_name} ready")

# Generate questions
if st.session_state.get("document_processed", False):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate Questions", type="primary"):
            with st.spinner("Creating questions..."):
                try:
                    import random
                    
                    # Get random chunks
                    total_chunks = len(st.session_state.vectordb.docstore._dict)
                    random_chunk_ids = random.sample(
                        list(st.session_state.vectordb.docstore._dict.keys()), 
                        min(10, total_chunks)
                    )
                    
                    random_chunks = []
                    for chunk_id in random_chunk_ids:
                        try:
                            chunk_content = st.session_state.vectordb.docstore._dict[chunk_id].page_content
                            random_chunks.append(chunk_content)
                        except:
                            continue
                    
                    if len(random_chunks) < 5:
                        sample_docs = st.session_state.vectordb.similarity_search("", k=20)
                        all_content = " ".join([doc.page_content for doc in sample_docs])
                        words = all_content.split()
                        meaningful_words = [w for w in words if len(w) > 4 and w.isalpha()]
                        
                        if len(meaningful_words) > 10:
                            search_terms = random.sample(meaningful_words, min(5, len(meaningful_words)))
                            retriever = st.session_state.vectordb.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 3}
                            )
                            
                            random_chunks = []
                            for term in search_terms:
                                try:
                                    docs = retriever.get_relevant_documents(term)
                                    random_chunks.extend([doc.page_content for doc in docs])
                                except:
                                    continue
                    
                    if len(random_chunks) > 8:
                        random_chunks = random.sample(random_chunks, 8)
                    
                    context = "\n\n".join(random_chunks)
                    
                    question_prompt = create_question_prompt()
                    llm = ChatOpenAI(
                        temperature=0.3,
                        openai_api_key=api_key,
                        model_name="gpt-4o-mini",
                        max_tokens=1500
                    )
                    
                    full_prompt = question_prompt.format(context=context[:6000])
                    response = llm.invoke(full_prompt)
                    
                    # Handle both string and AIMessage responses
                    if hasattr(response, 'content'):
                        st.session_state.generated_questions = response.content.strip()
                    else:
                        st.session_state.generated_questions = str(response).strip()
                    
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")

# Display questions
if st.session_state.get("generated_questions"):
    st.markdown("### Questions")
    st.markdown(st.session_state.generated_questions)
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download PDF"):
            try:
                pdf_buffer = create_pdf(st.session_state.generated_questions)
                st.download_button(
                    label="Click to Download",
                    data=pdf_buffer.getvalue(),
                    file_name="questions.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error("PDF creation failed")
