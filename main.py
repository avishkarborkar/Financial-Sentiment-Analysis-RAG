"""
Financial News Sentiment Analysis and RAG Pipeline

This module provides a comprehensive solution for analyzing financial news sentiment
and creating a retrieval-augmented generation (RAG) pipeline for financial Q&A.

Author: Avishkar Borkar
Date: 2024
"""

import os
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle

# Data processing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

# ML and NLP
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

# Financial analysis
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
from dataclasses import dataclass
from typing import Union

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for the financial analysis pipeline."""
    # File paths
    data_file: str = "financial_news_with_sentiment.csv"
    model_cache_dir: str = "./model_cache"
    vectorstore_path: str = "./vectorstore"
    
    # Model parameters
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-large"
    
    # Processing parameters
    batch_size: int = 32
    max_length: int = 512
    temperature: float = 0.1
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Financial parameters
    stock_symbols: List[str] = None
    lookback_days: int = 30
    
    def __post_init__(self):
        if self.stock_symbols is None:
            self.stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        # Create directories if they don't exist
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.vectorstore_path, exist_ok=True)


class DataValidator:
    """Validates and preprocesses financial data."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that the dataframe has required columns."""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (series < lower_bound) | (series > upper_bound)


class FinancialDataProcessor:
    """Handles loading and preprocessing of financial data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.validator = DataValidator()
        self.scaler = StandardScaler()
    
    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads and preprocesses the financial news data with comprehensive validation.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            logger.info(f"Loading data from {filepath}")
            
            # Load data with error handling
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records")
            
            # Validate required columns
            required_columns = ['headline', 'date', 'sentiment']
            self.validator.validate_dataframe(df, required_columns)
            
            # Clean and preprocess
            df = self._preprocess_dataframe(df)
            
            # Validate data quality
            self._validate_data_quality(df)
            
            logger.info("Data preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal method to preprocess the dataframe."""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean text columns
        text_columns = ['headline', 'content', 'summary']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.validator.clean_text)
        
        # Handle date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # Handle sentiment column
        if 'sentiment' in df.columns:
            # Normalize sentiment labels
            sentiment_mapping = {
                'positive': 'POSITIVE',
                'negative': 'NEGATIVE',
                'neutral': 'NEUTRAL',
                'pos': 'POSITIVE',
                'neg': 'NEGATIVE',
                'neu': 'NEUTRAL'
            }
            df['sentiment'] = df['sentiment'].str.lower().map(sentiment_mapping).fillna('NEUTRAL')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['headline'], keep='first')
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Validate data quality and log issues."""
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            logger.warning(f"Missing data found: {missing_data[missing_data > 0].to_dict()}")
        
        # Check for empty headlines
        empty_headlines = df['headline'].str.strip().eq('').sum()
        if empty_headlines > 0:
            logger.warning(f"Found {empty_headlines} empty headlines")
        
        # Check sentiment distribution
        if 'sentiment' in df.columns:
            sentiment_dist = df['sentiment'].value_counts()
            logger.info(f"Sentiment distribution: {sentiment_dist.to_dict()}")


class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple models and techniques."""
    
    def __init__(self, config: Config):
        self.config = config
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        self._load_models()
    
    def _load_models(self):
        """Load sentiment analysis models."""
        try:
            logger.info("Loading sentiment analysis models...")
            
            # Load the main sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
                tokenizer=self.config.sentiment_model,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("Sentiment models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {str(e)}")
            raise
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Perform batch sentiment analysis on a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment analysis results
        """
        try:
            if not texts:
                return []
            
            # Process in batches to avoid memory issues
            results = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_results = self.sentiment_pipeline(batch)
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            raise
    
    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive sentiment analysis on the dataframe.
        
        Args:
            df: DataFrame with 'headline' column
            
        Returns:
            DataFrame with sentiment analysis results
        """
        try:
            logger.info("Starting sentiment analysis...")
            
            # Get unique headlines to avoid duplicate processing
            unique_headlines = df['headline'].unique().tolist()
            logger.info(f"Analyzing sentiment for {len(unique_headlines)} unique headlines")
            
            # Perform batch sentiment analysis
            sentiment_results = self.analyze_sentiment_batch(unique_headlines)
            
            # Create mapping from headline to sentiment
            headline_to_sentiment = {}
            for headline, result in zip(unique_headlines, sentiment_results):
                # Get the highest confidence prediction
                best_prediction = max(result, key=lambda x: x['score'])
                headline_to_sentiment[headline] = {
                    'sentiment': best_prediction['label'],
                    'confidence': best_prediction['score']
                }
            
            # Apply sentiment to dataframe
            df['predicted_sentiment'] = df['headline'].map(lambda x: headline_to_sentiment[x]['sentiment'])
            df['sentiment_confidence'] = df['headline'].map(lambda x: headline_to_sentiment[x]['confidence'])
            
            # Add sentiment scores for numerical analysis
            sentiment_scores = {
                'POSITIVE': 1,
                'NEGATIVE': -1,
                'NEUTRAL': 0
            }
            df['sentiment_score'] = df['predicted_sentiment'].map(sentiment_scores)
            
            logger.info("Sentiment analysis completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise


class FinancialModeler:
    """Handles financial modeling and correlation analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.stock_data = {}
    
    def fetch_stock_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock market data for given symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
            
        Returns:
            Dictionary mapping symbols to stock data
        """
        try:
            logger.info(f"Fetching stock data for symbols: {symbols}")
            
            stock_data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    if not data.empty:
                        stock_data[symbol] = data
                        logger.info(f"Fetched {len(data)} records for {symbol}")
                    else:
                        logger.warning(f"No data found for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
            self.stock_data = stock_data
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise
    
    def correlate_sentiment_with_stocks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Correlate sentiment data with stock market performance.
        
        Args:
            df: DataFrame with sentiment data and dates
            
        Returns:
            Dictionary with correlation results
        """
        try:
            logger.info("Analyzing sentiment-stock correlations...")
            
            if not self.stock_data:
                logger.warning("No stock data available. Fetching data first...")
                self.fetch_stock_data(self.config.stock_symbols)
            
            results = {}
            
            # Group sentiment by date
            daily_sentiment = df.groupby(df['date'].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'sentiment_confidence': 'mean'
            }).reset_index()
            
            daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 'confidence_mean']
            
            for symbol, stock_df in self.stock_data.items():
                try:
                    # Calculate daily returns
                    stock_df['daily_return'] = stock_df['Close'].pct_change()
                    
                    # Merge with sentiment data
                    merged_data = pd.merge(
                        daily_sentiment,
                        stock_df[['Close', 'daily_return']].reset_index(),
                        left_on='date',
                        right_on='Date',
                        how='inner'
                    )
                    
                    if len(merged_data) > 10:  # Need sufficient data for correlation
                        # Calculate correlations
                        pearson_corr, pearson_p = pearsonr(
                            merged_data['sentiment_mean'],
                            merged_data['daily_return']
                        )
                        
                        spearman_corr, spearman_p = spearmanr(
                            merged_data['sentiment_mean'],
                            merged_data['daily_return']
                        )
                        
                        results[symbol] = {
                            'pearson_correlation': pearson_corr,
                            'pearson_p_value': pearson_p,
                            'spearman_correlation': spearman_corr,
                            'spearman_p_value': spearman_p,
                            'data_points': len(merged_data),
                            'mean_sentiment': merged_data['sentiment_mean'].mean(),
                            'mean_return': merged_data['daily_return'].mean()
                        }
                        
                        logger.info(f"Correlation for {symbol}: Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error calculating correlation for {symbol}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise
    
    def create_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators based on sentiment data.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            DataFrame with additional sentiment indicators
        """
        try:
            df = df.copy()
            
            # Sort by date
            df = df.sort_values('date')
            
            # Rolling averages
            df['sentiment_ma_7'] = df['sentiment_score'].rolling(window=7).mean()
            df['sentiment_ma_30'] = df['sentiment_score'].rolling(window=30).mean()
            
            # Sentiment momentum
            df['sentiment_momentum'] = df['sentiment_score'].diff()
            
            # Volatility
            df['sentiment_volatility'] = df['sentiment_score'].rolling(window=7).std()
            
            # Sentiment strength (absolute value)
            df['sentiment_strength'] = abs(df['sentiment_score'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating sentiment indicators: {str(e)}")
            raise


class AdvancedRAGPipeline:
    """Advanced RAG pipeline with improved document processing and retrieval."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG pipeline components."""
        try:
            logger.info("Initializing RAG pipeline components...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
            )
            
            logger.info("RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            raise
    
    def create_documents_from_dataframe(self, df: pd.DataFrame) -> List[Document]:
        """
        Create Document objects from dataframe with metadata.
        
        Args:
            df: DataFrame with financial news data
            
        Returns:
            List of Document objects
        """
        try:
            documents = []
            
            for _, row in df.iterrows():
                # Create document content
                content = f"Headline: {row['headline']}"
                
                if 'content' in row and pd.notna(row['content']):
                    content += f"\nContent: {row['content']}"
                
                if 'summary' in row and pd.notna(row['summary']):
                    content += f"\nSummary: {row['summary']}"
                
                # Create metadata
                metadata = {
                    'date': str(row['date']) if 'date' in row else None,
                    'sentiment': row.get('predicted_sentiment', 'UNKNOWN'),
                    'confidence': row.get('sentiment_confidence', 0.0),
                    'sentiment_score': row.get('sentiment_score', 0),
                    'source': 'financial_news'
                }
                
                # Create document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents from dataframe")
            return documents
            
        except Exception as e:
            logger.error(f"Error creating documents: {str(e)}")
            raise
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vectorstore from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vectorstore
        """
        try:
            logger.info("Creating vectorstore...")
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(split_docs)} chunks")
            
            # Create vectorstore
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            # Save vectorstore
            vectorstore.save_local(self.config.vectorstore_path)
            logger.info(f"Vectorstore saved to {self.config.vectorstore_path}")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {str(e)}")
            raise
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """Load existing vectorstore if available."""
        try:
            if os.path.exists(self.config.vectorstore_path):
                vectorstore = FAISS.load_local(
                    self.config.vectorstore_path,
                    self.embeddings
                )
                logger.info("Loaded existing vectorstore")
                return vectorstore
            return None
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore: {str(e)}")
            return None
    
    def create_qa_chain(self, vectorstore: FAISS) -> RetrievalQA:
        """
        Create QA chain with the vectorstore.
        
        Args:
            vectorstore: FAISS vectorstore
            
        Returns:
            RetrievalQA chain
        """
        try:
            logger.info("Creating QA chain...")
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create LLM
            llm = HuggingFaceHub(
                repo_id=self.config.llm_model,
                model_kwargs={
                    "temperature": self.config.temperature,
                    "max_length": self.config.max_length
                }
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    def create_rag_pipeline(self, df: pd.DataFrame) -> RetrievalQA:
        """
        Create complete RAG pipeline from dataframe.
        
        Args:
            df: DataFrame with financial news data
            
        Returns:
            RetrievalQA chain
        """
        try:
            # Try to load existing vectorstore
            vectorstore = self.load_vectorstore()
            
            if vectorstore is None:
                # Create new vectorstore
                documents = self.create_documents_from_dataframe(df)
                vectorstore = self.create_vectorstore(documents)
            
            # Create QA chain
            qa_chain = self.create_qa_chain(vectorstore)
            
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating RAG pipeline: {str(e)}")
            raise


class FinancialAnalysisPipeline:
    """Main pipeline class that orchestrates the entire analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = FinancialDataProcessor(config)
        self.sentiment_analyzer = AdvancedSentimentAnalyzer(config)
        self.financial_modeler = FinancialModeler(config)
        self.rag_pipeline = AdvancedRAGPipeline(config)
        
        # Results storage
        self.processed_data = None
        self.correlation_results = None
        self.qa_chain = None
    
    def run_complete_analysis(self, data_file: str) -> Dict[str, Any]:
        """
        Run the complete financial analysis pipeline.
        
        Args:
            data_file: Path to the financial news data file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info("Starting complete financial analysis pipeline...")
            
            results = {}
            
            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data...")
            self.processed_data = self.data_processor.load_and_preprocess_data(data_file)
            results['data_info'] = {
                'total_records': len(self.processed_data),
                'date_range': {
                    'start': str(self.processed_data['date'].min()),
                    'end': str(self.processed_data['date'].max())
                },
                'sentiment_distribution': self.processed_data['predicted_sentiment'].value_counts().to_dict()
            }
            
            # Step 2: Sentiment analysis
            logger.info("Step 2: Performing sentiment analysis...")
            self.processed_data = self.sentiment_analyzer.analyze_sentiment(self.processed_data)
            
            # Step 3: Financial modeling
            logger.info("Step 3: Performing financial modeling...")
            self.processed_data = self.financial_modeler.create_sentiment_indicators(self.processed_data)
            self.correlation_results = self.financial_modeler.correlate_sentiment_with_stocks(self.processed_data)
            results['correlation_analysis'] = self.correlation_results
            
            # Step 4: Create RAG pipeline
            logger.info("Step 4: Creating RAG pipeline...")
            self.qa_chain = self.rag_pipeline.create_rag_pipeline(self.processed_data)
            
            # Step 5: Generate sample insights
            logger.info("Step 5: Generating sample insights...")
            sample_queries = [
                "What is the overall market sentiment?",
                "What are the key negative news themes?",
                "How has sentiment changed over time?",
                "What stocks are most affected by sentiment?",
                "What are the main positive news drivers?"
            ]
            
            sample_responses = {}
            for query in sample_queries:
                try:
                    response = self.qa_chain({"query": query})
                    sample_responses[query] = {
                        'answer': response['result'],
                        'sources': len(response['source_documents'])
                    }
                except Exception as e:
                    logger.warning(f"Error processing query '{query}': {str(e)}")
                    sample_responses[query] = {'error': str(e)}
            
            results['sample_qa'] = sample_responses
            
            logger.info("Complete analysis pipeline finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: Question to ask
            
        Returns:
            Response from the QA chain
        """
        if self.qa_chain is None:
            raise ValueError("RAG pipeline not initialized. Run complete analysis first.")
        
        try:
            response = self.qa_chain({"query": question})
            return {
                'answer': response['result'],
                'sources': [
                    {
                        'content': doc.page_content[:200] + "...",
                        'metadata': doc.metadata
                    }
                    for doc in response['source_documents']
                ]
            }
        except Exception as e:
            logger.error(f"Error querying pipeline: {str(e)}")
            raise


def main():
    """Main function to run the improved financial analysis pipeline."""
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize pipeline
        pipeline = FinancialAnalysisPipeline(config)
        
        # Run complete analysis
        results = pipeline.run_complete_analysis(config.data_file)
        
        # Print results summary
        print("\n" + "="*50)
        print("FINANCIAL ANALYSIS RESULTS SUMMARY")
        print("="*50)
        
        print(f"\nData Information:")
        print(f"  Total records: {results['data_info']['total_records']}")
        print(f"  Date range: {results['data_info']['date_range']['start']} to {results['data_info']['date_range']['end']}")
        print(f"  Sentiment distribution: {results['data_info']['sentiment_distribution']}")
        
        print(f"\nCorrelation Analysis:")
        for symbol, corr_data in results['correlation_analysis'].items():
            print(f"  {symbol}: Pearson={corr_data['pearson_correlation']:.3f}, Spearman={corr_data['spearman_correlation']:.3f}")
        
        print(f"\nSample Q&A Responses:")
        for query, response in results['sample_qa'].items():
            print(f"\n  Q: {query}")
            if 'answer' in response:
                print(f"  A: {response['answer'][:200]}...")
            else:
                print(f"  A: Error - {response.get('error', 'Unknown error')}")
        
        # Interactive query loop
        print(f"\n" + "="*50)
        print("INTERACTIVE QUERY MODE")
        print("="*50)
        print("Enter your questions about the financial data (type 'quit' to exit):")
        
        while True:
            try:
                user_query = input("\nYour question: ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_query:
                    response = pipeline.query(user_query)
                    print(f"\nAnswer: {response['answer']}")
                    print(f"\nSources ({len(response['sources'])}):")
                    for i, source in enumerate(response['sources'][:3], 1):
                        print(f"  {i}. {source['content']}")
                        print(f"     Metadata: {source['metadata']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
        
        print("\nAnalysis complete. Thank you!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
