import gradio as gr

state = {"db": None}

def upload_pdf(pdf):
    state["db"] = build_index(pdf.name)
    return "Pdf indexed, ready for chat. ask questions about the PDF."

def chat(message, history):
    if state["db"] is None:
        return "Please upload a PDF first."
    return ask(state["db"], message)

with gr.Blocks(title="chat with your pdf") as demo:
    gr.Markdown("chat with your pdf (Powered by RAG)")
    pdf = gr.FIle(label="upload a pdf", file_types=[".pdf"])
    status = gr.Markdown()
    pdf.upload(upload_pdf, inputs=pdf, outputs=status)
    gr.ChatInterface(chat)

demo.launch(share=True)


def build_index(pdf_path):
    pages    = PyPDFLoader(pdf_path).load()                    # ① read all pages
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = splitter.split_documents(pages)                # ② chunk them
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db       = Chroma.from_documents(chunks, embedder)         # ③ in-memory store
    return db

