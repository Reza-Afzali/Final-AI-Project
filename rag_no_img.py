import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from dotenv import load_dotenv
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import CompositeElement
import hashlib

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"



load_dotenv()
api_key = os.getenv("API_KEY")
# Directories
DATASET_DIR = "Dataset"
CHROMA_DIR = "persist_store"



def generate_id(text):
        return hashlib.md5(text.encode()).hexdigest()


class RAGAgent:
    def __init__(self, data_dir="./Dataset", collection_name="ir_chunks",chroma_dir=CHROMA_DIR,gen_model="models/gemini-1.5-flash"):
        self.data_dir = data_dir
        #self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(gen_model)
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir, settings=Settings(allow_reset=True)
        )
        self.collection = self.chroma_client.get_or_create_collection(
                                                    name="sec_filings",
                                                    embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))
        #self.text_collection = self.chroma_client.get_or_create_collection(collection_name + "_text")
        #self.table_collection = self.chroma_client.get_or_create_collection(collection_name + "_table")

        
        
        self.embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") 
    

    
    def indexPDF(self):
        doc_count=0
        for company in os.listdir(self.data_dir):
            company_path = os.path.join(self.data_dir, company)
            if not os.path.isdir(company_path):
                continue

            for year in os.listdir(company_path):
                year_path = os.path.join(company_path, year)
                if not os.path.isdir(year_path):
                    continue

                for file in os.listdir(year_path):
                    if file.lower().endswith(".pdf"):
                        pdf_path = os.path.join(year_path, file)
                        print(f"📄 Processing: {pdf_path}")
                        existing = self.collection.query(
                        query_texts=["placeholder"],
                        n_results=1,
                        where={"source": file}
                         )

                        # Check if the first list inside `ids` has any results
                        if existing and existing["ids"] and existing["ids"][0]:
                            print(f"⏭️ Skipping already indexed file: {file}")
                            continue

                        try:
                            # Step 1: Parse the PDF using
                            elements = partition_pdf(
                                filename=pdf_path,
                                extract_tables=True,
                                strategy="auto"
                            )
                        except Exception as e:
                            print(f"❌ Failed to parse PDF {pdf_path}: {e}")
                            continue

                        try:
                            # Step 2: Chunk by title
                            title_chunks = chunk_by_title(elements)
                            print(f"✅ Found {len(title_chunks)} semantic chunks")

                            for chunk in title_chunks:
                                try:
                                    if isinstance(chunk, CompositeElement):
                                        chunk_text = chunk.text
                                    else:
                                        chunk_text = "\n\n".join(
                                            el.text for el in chunk
                                            if hasattr(el, 'text') and el.text and len(el.text) > 30
                                        )

                                    if chunk_text and len(chunk_text) > 30:
                                        self.collection.add(
                                            documents=[chunk_text],
                                            metadatas=[{
                                                "source": file,
                                                "company": company,
                                                "year": year
                                            }],
                                            ids=[generate_id(chunk_text)]
                                        )
                                        doc_count+=1
                                        
                                except Exception as e:
                                    print(f"⚠️ Failed to process a chunk in {file}: {e}")
                        except Exception as e:
                            print(f"❌ Chunking failed for {pdf_path}: {e}")

        print(f"✅ Indexed {doc_count} document chunks total.")

    
    
    def query(self, question, top_k=3):
        
        """
        Perform a semantic search over the vector store using the question embedding.
        
        Args:
            question (str): The user's input question.
            top_k (int): The number of top documents to retrieve.

        Returns:
            dict: A dictionary with keys "documents", "ids", and "metadatas".
        """
        # Perform query using the collection
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )

        return results

    
    def generate_answer(self, question, top_k=3):
        results = self.query(question, top_k=top_k)

        # Aggregate relevant chunks
        context_parts = results["documents"][0]
        context = "".join(context_parts)

        prompt = f"""You are a helpful assistant with access to corporate documents.
        Use the following context to answer the user's question:

    Context:
    {context}

    Question: {question}
    Answer:"""

        response = self.gemini_model.generate_content(prompt)

        # Generate citation list from metadata
        citations = []
        for metadata in results["metadatas"][0]:
            source = metadata.get("source", "Unknown source")
            company = metadata.get("company", "")
            year = metadata.get("year", "")
            citation = f"{source} ({company}, {year})"
            if citation not in citations:
                citations.append(citation)

        # Format citations
        citation_text = "Sources:" + " ".join(f"- {c}" for c in citations)

        return {
            "answer": response.text + citation_text,
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }


