import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rapidfuzz import fuzz, process
import re


def normalize_text(s):
    s=str(s).lower()
    s=re.sub(r'[^a-z0-9\s]','',s)
    s=re.sub(r'\s+','',s)
    return s.strip()



# Embeddings selection (try HF, fallback to legacy, then OpenAI)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


class HospitalDataLoader:
    def __init__(self, data_file):
        self.data_file = data_file
        self.vector_store = None
        self.df = None
        self.embeddings = embeddings

    def load_data(self):
        """Load CSV or return sample data if file missing/invalid."""
        try:
            if not os.path.exists(self.data_file):
                return self.create_sample_data()

            self.df = pd.read_csv(self.data_file)

            # Normalize common column names
            column_mapping = {
                "HOSPITAL NAME": "hospital_name",
                "Address": "address",
                "CITY": "city",
            }
            for old_name, new_name in column_mapping.items():
                if old_name in self.df.columns and new_name not in self.df.columns:
                    self.df.rename(columns={old_name: new_name}, inplace=True)

            required_columns = ["hospital_name", "city", "address"]
            if any(col not in self.df.columns for col in required_columns):
                return self.create_sample_data()

            if "network_status" not in self.df.columns:
                self.df["network_status"] = "In Network"

            return self.df
        except Exception:
            return self.create_sample_data()

    def create_sample_data(self):
        sample_data = {
            "hospital_name": [
                "Manipal Hospital Sarjapur",
                "Apollo Hospital Bannerghatta",
                "Fortis Hospital Cunningham Road",
                "Narayana Health Mazumdar Shaw",
                "Columbia Asia Hospital Yeshwanthpur",
            ],
            "city": ["Bangalore"] * 5,
            "address": [
                "Sarjapur Road, Bangalore, Karnataka",
                "Bannerghatta Road, Bangalore, Karnataka",
                "Cunningham Road, Bangalore, Karnataka",
                "Hosur Road, Bangalore, Karnataka",
                "Yeshwanthpur, Bangalore, Karnataka",
            ],
            "network_status": ["In Network", "In Network", "In Network", "Out of Network", "In Network"],
        }
        self.df = pd.DataFrame(sample_data)
        return self.df

    def create_documents(self, df):
        """Convert DataFrame rows to Document objects for vector indexing."""
        documents = []
        for _, row in df.iterrows():
            content = (
                f"Hospital: {row['hospital_name']}. "
                f"City: {row['city']}. "
                f"Address: {row['address']}. "
                f"Network Status: {row.get('network_status', '')}."
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "hospital_name": row.get("hospital_name", ""),
                        "city": row.get("city", ""),
                        "address": row.get("address", ""),
                        "network_status": row.get("network_status", ""),
                    },
                )
            )
        return documents

    def create_vector_store(self, documents):
        """Build FAISS vector store from documents."""
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return self.vector_store
        except Exception:
            return None

    def search_hospitals(self, query, k=5):
        """Semantic search (returns list of Document objects)."""
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception:
            return []

    def exact_match_search(self, hospital_name=None, city=None, fuzzy_thresh=85, max_fuzzy=5):

        results=self.df.copy()

        if "hospital_name_norm" not in results.columns:
            results["hospital_name_norm"]=results["hospital_name"].astype(str).apply(normalize_text)


        q=normalize_text(hospital_name)

        #word boundary regex
        pattern =r'\b'+re.escape(q)+r'\b'
        exact=results[results["hospital_name_norm"].str.contains(pattern,regex=True)]
        if not exact.empty:
            return exact
        
        #check substring
        sub=results[results["hospital_name_norm"].str.contains(q)]
        if not sub.empty: 
            return sub
        
        choices=results["hospital_name_norm"].tolist()
        fuzzy=process.extract(q, choices,scorer=fuzz.WRatio,limit=5)
        good=[c for c,s,i in fuzzy if s>=85]
        return results[results["hospital_name_norm"].isin(good)]
    


        
