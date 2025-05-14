import scrapper

from common import Data
import chromadb
import os
import datetime
import json
from chromadb.config import Settings
import time
from datetime import datetime
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np

from langchain_core.messages import  SystemMessage
from langchain.prompts import ChatPromptTemplate
from  langchain_core.output_parsers.string import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
STANDARD_MODEL = "gpt-3.5-turbo"
QUICK_MODEL = "gemini-2.0-flash-lite"
REASONING_MODEL = "o3-mini"


class SummaryAgent:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
           "key.json"
        )
        # Initialize language model
        model = ChatGoogleGenerativeAI(
            model=QUICK_MODEL,
            temperature=0.1
        )

        # Add custom instructions to the prompt template
        additional_instructions = """
            Hãy phân tách phần "Nội dung" thành các mệnh đề rõ ràng và đơn giản, đảm bảo rằng chúng có thể hiểu được ngay cả khi tách khỏi ngữ cảnh.
Tách câu ghép thành các câu đơn. Giữ nguyên cách diễn đạt ban đầu trong đầu vào bất cứ khi nào có thể.
Với bất kỳ thực thể nào có tên riêng đi kèm với thông tin mô tả bổ sung, hãy tách thông tin này thành một mệnh đề riêng biệt.
Phi ngữ cảnh hóa các mệnh đề bằng cách thêm các từ bổ nghĩa cần thiết vào danh từ hoặc toàn bộ câu, và thay thế các đại từ (ví dụ: "nó", "anh ấy", "cô ấy", "họ", "điều này", "điều đó") bằng tên đầy đủ của thực thể mà chúng đề cập đến.
 Dựa trên thông tin được cung cấp, hãy tóm tắt thành một mảng JSON.
 Trả về đúng định dạng JSON array.
            Ví dụ JSON: ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3"]
Ví dụ trả lời:
"Nội dung": "Theo các chuyên gia của Công ty cổ phần Chứng khoán SHS, xu hướng ngắn hạn VN-Index đang chuyển sang giai đoạn tích lũy sau khi vượt lên xu hướng giảm giá ngắn hạn. VN-Index cũng vượt lên các vùng kháng cự quan trọng, giá trung bình 200 phiên quanh 1.265 điểm và giá trung bình 200 tuần quanh 1.235 điểm.
Tâm lý và xu hướng của thị trường đang cải thiện. Chỉ số VN30 cũng tích cực khi vượt lên giá của phiên giảm mạnh trước thông tin áp thuế. Tuy nhiên dự kiến VN30 có thể chịu áp lực điều chỉnh ngắn hạn khi tiếp tục hướng đến vùng giá quanh 1.360 điểm -1.370 điểm. VN-Index gặp vùng kháng cự 1.275 điểm, vùng giá cao của phiên giảm mạnh 3/4/2025.
"Thị trường đang giai đoạn phục hồi, tạo vùng giá cân bằng với nhiều mã, nhóm mã sau giai đoạn giảm sốc. Với những thông tin đàm phán có thể thuế đối ứng sẽ giảm. Tuy nhiên kết quả như thế nào thì mức thuế quan cũng sẽ được áp đặt, cán cân thương mại thặng dư với Mỹ sẽ chịu áp lực mạnh. Ảnh hướng đến các cân đối vĩ mô, hoạt động sản xuất, kinh doanh của doanh nghiệp", nhóm phân tích SHS nhận định."
"Kết quả": [
  "Các chuyên gia của Công ty Chứng khoán SHS nhận định xu hướng ngắn hạn của VN-Index đang chuyển sang giai đoạn tích lũy.",
  "VN-Index đã vượt lên xu hướng giảm giá ngắn hạn.",
  "VN-Index đã vượt các vùng kháng cự quan trọng.",
  "VN-Index đã vượt giá trung bình 200 phiên quanh mức 1.265 điểm.",
  "VN-Index đã vượt giá trung bình 200 tuần quanh mức 1.235 điểm.",
  "Tâm lý thị trường đang cải thiện.",
  "Xu hướng thị trường đang cải thiện.",
  "Chỉ số VN30 tích cực khi vượt giá của phiên giảm mạnh do thông tin áp thuế.",
  "VN30 có thể chịu áp lực điều chỉnh ngắn hạn.",
  "VN30 đang hướng đến vùng giá 1.360 đến 1.370 điểm.",
  "VN-Index đang gặp vùng kháng cự 1.275 điểm.",
  "Mức 1.275 điểm là vùng giá cao của phiên giảm mạnh ngày 3 tháng 4 năm 2025.",
  "Thị trường đang trong giai đoạn phục hồi.",
  "Thị trường đang tạo vùng giá cân bằng sau giai đoạn giảm sốc.",
  "Có thông tin đàm phán cho thấy thuế đối ứng có thể sẽ giảm.",
  "Dù kết quả đàm phán thế nào, mức thuế quan vẫn sẽ được áp đặt.",
  "Cán cân thương mại thặng dư với Mỹ sẽ chịu áp lực mạnh.",
  "Các yếu tố vĩ mô sẽ chịu ảnh hưởng.",
  "Hoạt động sản xuất và kinh doanh của doanh nghiệp sẽ bị ảnh hưởng."]
            """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=additional_instructions),
            ("user", "{input}"),
        ])
        self.llm = prompt | model | StrOutputParser()


class VietnameseBiEncoderEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
        
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
            
        # Tạo embeddings
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()

def convert_date_to_iso(date_str: str) -> float:
        # Thử parse với định dạng DD/MM/YYYY
        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        return date_obj.timestamp()

def main():
    print("Data before splitting:")
    chroma_path = "./data/chroma_db"
    os.makedirs(chroma_path, exist_ok=True)
    
    # Tạo embedding function với model Vietnamese Bi-Encoder
    embedding_function = VietnameseBiEncoderEmbeddingFunction()
    
    client = chromadb.PersistentClient(
        path=chroma_path, 
        settings=Settings(allow_reset=True, is_persistent=True)
    )
    data = scrapper.scrapper()
    
    # Tạo collection với embedding function tùy chỉnh
    collection = client.get_or_create_collection(
        name="financial_news",
        embedding_function=embedding_function
    )
    count = collection.count()

    print(f"Dữ liệu thu thập được có : {count} bản ghi")
    agent = SummaryAgent()
    result = []
    for idx, d in enumerate(data):
        content_text = d.content
        print(f"Adding document {idx}/{len(data)}")
        message = [("human", content_text)]

        text_summary = agent.llm.invoke(message)
        # Xử lý chuỗi JSON
        text_summary = text_summary.strip()
        text_summary = text_summary.replace("json", "")
        if text_summary.startswith("```"):
            text_summary = text_summary[3:]
        if text_summary.endswith("```"):
            text_summary = text_summary[:-3]
        text_summary = text_summary.strip()
        
        try:
            parsed_array = json.loads(text_summary)
        except json.JSONDecodeError as e:
            print(f"message: {message}")
            print(f"Lỗi parse JSON: {e}")
            print(f"Text gốc: {text_summary}")
            parsed_array = [text_summary]  # Fallback to original text

        iso_date = convert_date_to_iso(d.date)
        # print(f"Summary for {iso_date}: {text_summary}")
        for item in parsed_array:
            result.append(Data(iso_date, item))
        time.sleep(2.0)
    print(f"Adding {len(result)} documents to ChromaDB...")
    for i, d in enumerate(result, 1):
        # Store data in ChromaDB with metadata
        collection.add(
            documents=[d.content],
            ids=[f"doc_{i}"],
            metadatas=[{"date": d.date}]  # Đã được chuyển đổi sang ISO format
        )
    print("All documents have been added to ChromaDB successfully!")

if __name__ == "__main__":
    main()
