import fitz #PyMuPdf
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np 
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# CLIP Model
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API KEY")


# intitializing the CLIP Model for unified embeddings

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Embedding Images

def embed_image(image_data):
    """ Embed Image using CLIP """
    if isinstance(image_data,str): #if path
        image = Image.open(image_data).convert("RGB") 
    else: # if PIL Image
        image = image_data

    inputs = clip_processor(images=image,return_tensors="pt") # return image data in pytorch tensors
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Normalizing Embeddings to unit vector.\
        features = features / features.norm(dim=-1,keepdim=True)
        return features.squeeze().numpy()


def embed_text(text):
    """ Embed text using CLIP """
    inputs = clip_processor(
        text = text,
        return_tensors = 'pt',
        padding = True,
        truncation = True,
        max_length=77 # CLIP's max token length
    ) 
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        # Normalizing Embeddings to unit vector.\
        features = features / features.norm(dim=-1,keepdim=True)
        return features.squeeze().numpy()


 ## Process PDF
pdf_path = "Company Profile_compressed.pdf"
doc = fitz.open(pdf_path)
# Storage for all documents and embeddings

all_docs = []
all_embeddings = []
image_data_store = {} # Stores Actual Image data for LLM

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

for i, page in enumerate(doc): 
    # Process Text
    text = page.get_text()
    if text.strip():
        ## Create temporary document for splitting
        temp_doc = Document(page_content=text,metadata={"page":i,"type" : "text"})
        text_chunks = splitter.split_documents([temp_doc])

        # Embed Each chunks using CLIP
        for chunk in text_chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)


    # Three Important steps for image : Convert pdf image to PIL Format, Store as base64 for GPT-4V(Which means base64 images), Create CLIP embedding for retrieval

    for img_index, img in enumerate(page.get_images(full=True)):
        try: 
            xref = img[0]
            base_img = doc.extract_image(xref)
            image_bytes = base_img["image"]

            # convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Create unique identifier
            image_id = f"page_{i}_img_{img_index}"

            # Storing image as base64 for later use with GPT-4V
            buffered = io.BytesIO()
            pil_image.save(buffered,format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64

            # Embed image using CLIP
            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)

            # Create documents for image
            image_doc = Document(
                page_content = f"[image : {image_id}]",
                metadata={"page":i,"type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)

        except Exception as e:
            print(f"Error Processing Image {img_index} on page {i}:{e}")
            continue

doc.close()


# Creating unified FAISS Vector store with CLIP embeddings

embeddings_array = np.array(all_embeddings)

# Create Custom FAISS index since we have precomputed embeddings
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,  # We're using precomputed embeddings
    metadatas=[doc.metadata for doc in all_docs]
)

llm = init_chat_model("openai:gpt-4.1")

def retrieve_multimodal(query,k=5):
    """ Unified retreival using CLIP embeddings for both text and images. """

    # Embed query using CLIP
    query_embedding = embed_text(query)

    # search in unified vector Store

    results = vector_store.similarity_search_by_vector(
        embedding = query_embedding,
        k=k
    )

    return results

def create_multimodal_message(query,retrieved_docs):
    """ Create a message with both text and images for GPT-4v """

    content = []

    # Add the Query
    content.append({
        "type" : "text",
        "text" : f"Question: {query}\n\n context : \n"}
    )

    # Seperate text and image documents

    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_doc = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]


    # Add text Context

    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
        ])

        content.append({"type" : "text",
        "text" : f"Text Excerpts: \n{text_context}\n"})

    # Add Images

    for doc in image_doc:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text" : f"\n[Image from Page {doc.metadata['page']}]:\n"
            })
    content.append({
        "type" : "image_url",
        "image_url" : {
            "url": f"data:image/png;base64,{image_data_store[image_id]}"
        }
    })

    return HumanMessage(content=content)

    # Add Instruction

    content.append({
        "type" : "text",
        "text" : "\n\nPlease answer the question based on the provided text and images."
    })

    
def multimodal_pdf_rag_pipeline(query):
    """Main pipeline for multimodal RAG."""
    # Retrieve relevant documents
    context_docs = retrieve_multimodal(query, k=5)
    
    # Create multimodal message
    message = create_multimodal_message(query, context_docs)
    
    # Get response from GPT-4V
    response = llm.invoke([message])
    
    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")
    
    return response.content

if __name__ == "__main__":
    # Example queries
    queries = [
        "What does the chart on page 1 show about revenue trends?",
        "Summarize the main findings from the document",
        "What visual elements are present in the document?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(query)
        print(f"Answer: {answer}")
        print("=" * 70)  