from dotenv import load_dotenv
load_dotenv()

from openai	import OpenAI

# API 키 입력
client = OpenAI()

# 생성할 파일명
speech_file_path = "speech.mp3"

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="""오늘은 사람들이 좋아하는 것을 만들기에 좋은 날입니다!""",
) as response:
    response.stream_to_file("speech.mp3")