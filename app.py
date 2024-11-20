import os
import sys
from distutils.util import strtobool

import boto3
from dotenv import load_dotenv
from flask import Flask, abort, request
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.pgvector import PGVector
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from opencc import OpenCC

import src.constants as const
import src.utils as utils
from src.utils import MaxPoolingEmbeddings, PathHelper, get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Get LINE Bot credentials
channel_secret = os.getenv("CHANNEL_SECRET")
channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
if not channel_secret or not channel_access_token:
    logger.error("Missing LINE_CHANNEL_SECRET or LINE_CHANNEL_ACCESS_TOKEN.")
    sys.exit("Environment variables missing. Check your .env file.")

# Support multi-lingual
support_multilingual = strtobool(os.getenv("SUPPORT_MULTILINGUAL", "False"))

# Configure retriever
def configure_retriever():
    try:
        logger.info("Configuring retriever...")
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
        logger.info("Retriever configured successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Error configuring retriever: {e}")
        sys.exit("Failed to configure retriever.")

# Initialize Flask app
app = Flask(__name__)
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# Initialize LLM
llm = ChatOpenAI(
    model_name=const.CHAT_GPT_MODEL_NAME,
    temperature=0,
    streaming=True,
)

# Initialize retriever
retriever = configure_retriever()

# Simplified Chinese to Traditional Chinese converter
s2t_converter = OpenCC("s2t")

# AWS services for translation and language detection
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

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)

# LangChain QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
    return_source_documents=True,
)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error(f"Error handling LINE webhook: {e}")
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    question = event.message.text.strip()

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
            if support_multilingual:
                question_lang_obj = comprehend.detect_dominant_language(Text=question)
                question_lang = question_lang_obj["Languages"][0]["LanguageCode"]
            else:
                question_lang = const.DEFAULT_LANG

            response = qa_chain({"question": question})
            answer = response["answer"]
            answer = s2t_converter.convert(answer)

            if support_multilingual and question_lang != "zh-TW":
                answer_translated = translate.translate_text(
                    Text=answer,
                    SourceLanguageCode="zh-TW",
                    TargetLanguageCode=question_lang,
                )
                answer = answer_translated["TranslatedText"]

            ref_video_template = ""
            if "source_documents" in response:
                for i in range(min(const.N_SOURCE_DOCS, len(response["source_documents"]))):
                    doc = response["source_documents"][i]
                    video_id = doc.metadata.get("video_id", "unknown")
                    if video_id != "unknown":
                        url = f"https://www.youtube.com/watch?v={video_id}"
                        ref_video_template += f"{url}\n"
            answer += f"\n\nSource:\n{ref_video_template}" if ref_video_template else ""
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "抱歉，我無法處理您的請求，請稍後再試。"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
