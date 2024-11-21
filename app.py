import os
import sys
from distutils.util import strtobool
from flask import Flask, request, abort
from linebot.v3.messaging import MessagingApi as LineBotApi
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from opencc import OpenCC
import boto3
import src.constants as const
import src.utils as utils
from src.utils import MaxPoolingEmbeddings, PathHelper, get_logger

# 初始化 logger
logger = get_logger(__name__)

# 載入環境變數
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# 取得 LINE Bot 憑據
channel_secret = os.getenv("CHANNEL_SECRET")
channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
if not channel_secret or not channel_access_token:
    logger.error("缺少 CHANNEL_SECRET 或 CHANNEL_ACCESS_TOKEN.")
    sys.exit("環境變數缺失，請檢查 .env 檔案。")

# 初始化 Flask 應用程式
app = Flask(__name__)

# 初始化 LINE Bot API 和 Webhook Handler
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# 是否支援多語言
support_multilingual = strtobool(os.getenv("SUPPORT_MULTILINGUAL", "False"))

# 配置檢索器
def configure_retriever():
    try:
        logger.info("正在配置檢索器...")
        embeddings = MaxPoolingEmbeddings(
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_name=const.ENCODING_MODEL_NAME,
        )
        vectordb = PGVector(
            collection_name=const.COLLECTION_NAME,
            connection_string=utils.get_connection_string(),
            embedding_function=embeddings,
        )
        retriever = vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": const.N_DOCS}
        )
        logger.info("檢索器配置成功。")
        return retriever
    except Exception as e:
        logger.error(f"配置檢索器時發生錯誤: {e}")
        sys.exit("檢索器配置失敗。")

# 初始化 LLM
llm = ChatOpenAI(
    model_name=const.CHAT_GPT_MODEL_NAME,
    temperature=0,
    streaming=True,
)

# 初始化檢索器
retriever = configure_retriever()

# 簡體轉繁體轉換器
s2t_converter = OpenCC("s2t")

# 初始化 AWS 服務
translate = boto3.client(
    "translate",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME"),
)
comprehend = boto3.client(
    "comprehend",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME"),
)

# 初始化記憶體
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)

# LangChain QA 鏈
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
    return_source_documents=True,
)

# LINE Webhook 回調函數
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    logger.info(f"請求內容: {body}")

    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error(f"處理 LINE Webhook 時發生錯誤: {e}")
        abort(500)

    return "OK", 200

# 處理 LINE 訊息事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    question = event.message.text.strip()
    logger.info(f"收到的訊息: {question}")  # 日誌記錄收到的訊息

    if question.startswith("/清除") or question.lower().startswith("/clear"):
        memory.clear()
        answer = "歷史訊息清除成功。"
    elif question.lower().startswith("/help") or question.startswith("/說明"):
        answer = (
            "指令：\n"
            "/清除 或 /clear - 清除歷史對話。\n"
            "/說明 或 /help - 查看指令幫助。"
        )
    else:
        try:
            # 這裡加入日誌，檢查是否有進行回答生成
            logger.info("正在生成回答...")
            response = qa_chain({"question": question})
            answer = response["answer"]
            logger.info(f"生成的回答: {answer}")  # 日誌記錄生成的回答

        except Exception as e:
            logger.error(f"回答生成時出錯: {e}")
            answer = "抱歉，無法處理您的請求。"

    # 確保 reply_message 正確調用
    try:
        logger.info(f"準備回覆: {answer}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
        logger.info("回覆訊息已發送")
    except Exception as e:
        logger.error(f"回覆訊息時發生錯誤: {e}")
# 主程式運行
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
