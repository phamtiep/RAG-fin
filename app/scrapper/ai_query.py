import scrapper


import chromadb
import os

from chromadb.config import Settings
import time
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from  langchain_core.output_parsers.string import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

STANDARD_MODEL = "gpt-3.5-turbo"
QUICK_MODEL = "gemini-2.0-flash-lite"
REASONING_MODEL = "o3-mini"

# print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
# print(f"LANGCHAIN_API_KEY: {os.getenv('LANGCHAIN_API_KEY')}")

STANDARD_MODEL = "gpt-3.5-turbo"
QUICK_MODEL = "gemini-2.0-flash-lite"
REASONING_MODEL = "o3-mini"
def get_date_extraction_prompt():
    import datetime
    
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    
    # Tính toán các ngày tham chiếu khác
    seven_days_ago = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    month_start = datetime.datetime(now.year, now.month, 1).strftime("%Y-%m-%d")
    quarter_start = datetime.datetime(now.year, ((now.month-1)//3)*3+1, 1).strftime("%Y-%m-%d")
    year_start = datetime.datetime(now.year, 1, 1).strftime("%Y-%m-%d")
    
    last_week_start = (now - datetime.timedelta(days=now.weekday()+7)).strftime("%Y-%m-%d")
    last_week_end = (now - datetime.timedelta(days=now.weekday()+1)).strftime("%Y-%m-%d")
    
    last_month = now.month-1 if now.month > 1 else 12
    last_month_year = now.year if now.month > 1 else now.year-1
    last_month_start = datetime.datetime(last_month_year, last_month, 1).strftime("%Y-%m-%d")
    last_month_end = (datetime.datetime(now.year, now.month, 1) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    q1_start = datetime.datetime(now.year, 1, 1).strftime("%Y-%m-%d")
    q1_end = datetime.datetime(now.year, 3, 31).strftime("%Y-%m-%d")
    
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are an assistant specialized in extracting dates from user questions.
Return dates in YYYY-MM-DD format.
If there's a date range, return as: start_date|end_date
If no explicit dates found, infer appropriate date ranges for time-sensitive queries.
Rules for Date Extraction:
For explicit dates, convert to YYYY-MM-DD format
For date ranges, return as start_date|end_date
For relative time references, calculate based on current date ({current_date})
For vague time references, infer appropriate date ranges
Examples in Vietnamese:
"Tin tức mới nhất" -> "{seven_days_ago}|{current_date}" (7 ngày gần đây)
"Sự kiện gần đây" -> "{seven_days_ago}|{current_date}" (7 ngày gần đây)
"Tin tức trong ngày" -> "{current_date}|{current_date}" (chỉ hôm nay)
"Tin tức tuần này" -> "{seven_days_ago}|{current_date}" (tuần hiện tại)
"Tin tức tháng này" -> "{month_start}|{current_date}" (tháng hiện tại)
"Tin tức quý này" -> "{quarter_start}|{current_date}" (quý hiện tại)
"Tin tức năm nay" -> "{year_start}|{current_date}" (năm hiện tại)
"Tin tức tuần trước" -> "{last_week_start}|{last_week_end}" (tuần trước)
"Tin tức tháng trước" -> "{last_month_start}|{last_month_end}" (tháng trước)
"Tin tức quý 1" -> "{q1_start}|{q1_end}" (quý 1 năm hiện tại)
Always return ONLY the date format without explanation. If you need to infer dates, choose appropriate ranges based on the context and current date.
        """),
        ("user", "{input}")
    ])

class VietnameseBiEncoderEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
        
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
            
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()

class QueryAgent:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
        
        # Initialize model for date extraction
        date_extraction_model = ChatGoogleGenerativeAI(
            model=QUICK_MODEL,
            temperature=0.1  # Lower temperature for more precise results
        )
        
        # Prompt template for date extraction

        date_extraction_prompt = get_date_extraction_prompt()
        self.date_extractor = date_extraction_prompt | date_extraction_model | StrOutputParser()
        
        # Initialize model for answering questions
        model = ChatGoogleGenerativeAI(
            model=QUICK_MODEL,
            temperature=0.7
        )

        # Prompt template for answering questions
        additional_instructions = (
            "Your task is to answer all the questions that users ask based only on the context information provided below.\n"
            "Please answer the question at length and in detail, with full meaning.\n"
            "In the answer there is no sentence such as: based on the context provided.\n"
            "Your response should be in Vietnamese.\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        prompt = ChatPromptTemplate.from_messages([
            additional_instructions
        ])
        
        chroma_path = "./data/chroma_db"
        embedding_function = VietnameseBiEncoderEmbeddingFunction()
        
        client = chromadb.PersistentClient(
            path=chroma_path, 
            settings=Settings(allow_reset=True, is_persistent=True)
        )
        
        self.collection = client.get_or_create_collection(
            name="financial_news",
            embedding_function=embedding_function
        )
        count = self.collection.count()
        print(f"Dữ liệu thu thập được có : {count} bản ghi")
        self.llm = prompt | model | StrOutputParser()
        
        additional_remove_time = (
           "Remove any temporal (time-related) context from the following question. This includes words or phrases indicating time such as specific dates, times of day, relative time expressions (e.g., yesterday, next week, in 2025), or any reference to timeframes."
            "Return only the modified question with the temporal context removed."
            "Do not provide any explanation, comments, or extra output — return only the cleaned question."   
            "Example Input:"
            "Thị trường chứng khoán hôm nay thế nào ?"
            "Expected Output:"
            "Thị trường chứng khoán thế nào ?"
            "Input question: {question}"
        )
        prompt_remove_time = ChatPromptTemplate.from_messages([
            additional_remove_time
        ])
        self.llm_remove_time = prompt_remove_time | model | StrOutputParser()

    def extract_dates(self, question: str) -> tuple:
        """Extract dates from user's question"""
        date_result = self.date_extractor.invoke({"input": question})
        if date_result == "none":
            return None, None
        
        dates = date_result.split("|")
        if len(dates) == 2:
            return dates[0], dates[1]
        return dates[0], None

    def convert_date_to_timestamp(self, date_str: str) -> float:
        """Chuyển đổi chuỗi ngày tháng sang timestamp (số giây từ epoch)"""
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return float(date_obj.timestamp())
    def remove_time_query(self, question: str) -> str:
        """Xóa các từ khóa liên quan đến thời gian trong câu hỏi"""
        time_keywords = [
            "hôm nay", "ngày hôm nay", "tuần này", "tháng này", "quý này", "năm nay",
            "tuần trước", "tháng trước", "quý trước", "năm trước",
            "tuần sau", "tháng sau", "quý sau", "năm sau"
        ]
        for keyword in time_keywords:
            question = question.replace(keyword, "")
        return question.strip()
    
    def query(self, question: str) -> str:
        start_date, end_date = self.extract_dates(question)
        print(start_date, end_date)
        where = None
        if start_date is None and end_date:
            start_date = end_date
        if end_date is None and start_date:
            end_date = start_date
        if start_date and end_date:
            # Chuyển đổi ngày tháng sang timestamp
            start_timestamp = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
            end_timestamp = datetime.strptime(end_date, "%Y-%m-%d").timestamp()
            
        where = {
            "$and": [
                {"date": {"$gte": start_timestamp}},
                {"date": {"$lte": end_timestamp}}
            ]
        }

        results = self.collection.query(
            query_texts=[question],
            n_results=100,
            where=where
        )
        print(where)
        print(results)
        
        context = ""
        for i in range(len(results["documents"])):
            context += str(results["documents"][i]) + "\n"
        context_str = context  
        query_str = question
        new_question = self.llm_remove_time.invoke({"question": query_str})
        text_summary = self.llm.invoke({"context_str": context_str, "query_str": new_question})
        return text_summary


