# constants
VIDEO_ID = "video_id"
VIDEO_URL = "video_url"
TITLE = "title"
TRANSCRIPT = "transcript"  # 修正拼字
CHANNEL_NAME = "channel_name"

# model name
# 由於需要切換到 GPT-4，更新模型名稱
ENCODING_MODEL_NAME = "GanymedeNil/text2vec-large-chinese"  # 編碼模型保持不變
CHAT_GPT_MODEL_NAME = "gpt-4"  # 修改為 GPT-4

# db
DB = "db"
CHROMA_DB = "chroma"

# m docs
N_DOCS = 5
N_SOURCE_DOCS = 2

# message
INIT_MESSAGE = "我可以怎麼幫你呢？"
TEST_RESPONSES = ["測試一", "測試二", "測試三", "測試四", "測試五"]

# condense_question_prompt
PROMPT = (
    "Given the following conversation and a follow-up question, "
    + "rephrase the follow-up question to be a standalone question, "
    + "in Traditional Chinese and DO NOT in Simplified Chinese."
)

# lang
DEFAULT_LANG = "zh-TW"

# s3
S3_BUCKET_NAME = "ask-me-parenting"

# collection
COLLECTION_NAME = "video_chunks"
