import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Using HuggingFace embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        print("✅ Using legacy HuggingFace embeddings")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        # Fallback to OpenAI if nothing else works
        from langchain_openai import OpenAIEmbeddings
        print("⚠️ Using OpenAI embeddings (may incur costs)")
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

class HospitalDataLoader:
    def __init__(self, data_file):
        self.data_file = data_file
        self.vector_store = None
        self.df = None
        self.embeddings = embeddings
        
    def load_data(self):
        """Load and validate hospital data"""
        try:
            if not os.path.exists(self.data_file):
                print(f"❌ CSV file not found: {self.data_file}")
                print("📋 Creating sample data for testing...")
                return self.create_sample_data()
            
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(self.df)} records from {self.data_file}")
            
            # Map your CSV columns to expected names
            column_mapping = {
                'HOSPITAL NAME': 'hospital_name',
                'Address': 'address', 
                'CITY': 'city'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in self.df.columns and new_name not in self.df.columns:
                    self.df.rename(columns={old_name: new_name}, inplace=True)
                    print(f"✅ Renamed column: {old_name} -> {new_name}")
            
            # Validate required columns
            required_columns = ['hospital_name', 'city', 'address']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                print("📋 Using sample data instead")
                return self.create_sample_data()
            
            # Add network_status if missing
            if 'network_status' not in self.df.columns:
                self.df['network_status'] = 'In Network'
                print("✅ Added 'network_status' column")
                
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create comprehensive sample data"""
        sample_data = {
            'hospital_name': [
                'Manipal Hospital Sarjapur', 
                'Apollo Hospital Bannerghatta', 
                'Fortis Hospital Cunningham Road',
                'Narayana Health Mazumdar Shaw',
                'Columbia Asia Hospital Yeshwanthpur'
            ],
            'city': ['Bangalore', 'Bangalore', 'Bangalore', 'Bangalore', 'Bangalore'],
            'address': [
                'Sarjapur Road, Bangalore, Karnataka',
                'Bannerghatta Road, Bangalore, Karnataka', 
                'Cunningham Road, Bangalore, Karnataka',
                'Hosur Road, Bangalore, Karnataka',
                'Yeshwanthpur, Bangalore, Karnataka'
            ],
            'network_status': ['In Network', 'In Network', 'In Network', 'Out of Network', 'In Network']
        }
        self.df = pd.DataFrame(sample_data)
        print("📋 Using sample hospital data")
        return self.df
    
    def create_documents(self, df):
        """Convert DataFrame to searchable documents"""
        documents = []
        for _, row in df.iterrows():
            content = f"Hospital: {row['hospital_name']}. City: {row['city']}. Address: {row['address']}. Network Status: {row['network_status']}."
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'hospital_name': row['hospital_name'],
                    'city': row['city'],
                    'address': row['address'],
                    'network_status': row['network_status']
                }
            ))
        print(f"✅ Created {len(documents)} searchable documents")
        return documents
    
    def create_vector_store(self, documents):
        """Create FAISS vector database"""
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("✅ Vector database created successfully")
            return self.vector_store
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return None
    
    def search_hospitals(self, query, k=5):
        """Semantic search using vector database"""
        if self.vector_store is None:
            print("❌ Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f"🔍 Semantic search found {len(results)} hospitals for: '{query}'")
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def exact_match_search(self, hospital_name=None, city=None):
        """Exact match search for network verification"""
        if self.df is None:
            self.load_data()
        
        results = self.df.copy()
        
        if hospital_name:
            results = results[results['hospital_name'].str.contains(hospital_name, case=False, na=False)]
            print(f"🔍 Exact match for hospital: '{hospital_name}'")
        
        if city:
            results = results[results['city'].str.contains(city, case=False, na=False)]
            print(f"🔍 Exact match for city: '{city}'")
        
        print(f"✅ Exact match found {len(results)} results")
        return results