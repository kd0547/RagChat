import base64
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama


class OllamaModel:

    def __init__(self,
                 model_name:str,
                 num_ctx:int = 8192,
                 num_gpu:int = -1,
                 num_thread:int = 4,
                 temperature:float=0.8,
                 reasoning: bool | str | None = None,
                 system_prompt: str | None = None,
                 ):
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.temperature = temperature
        self.reasoning = reasoning
        self.system_prompt = system_prompt

        self.llm = ChatOllama(
            model= model_name,
            num_gpu=num_gpu,
            num_thread=num_thread,
            temperature=temperature,
            reasoning=reasoning
        )

        if self.system_prompt is None:
            system_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])

    def invoke_image(self,
                     image_bytes:bytes,
                     prompt: str,
                     encoding:str = "utf-8"):
        """
        멀티모달 (이미지 + 텍스트) 입력
        Ollama의 vision 모델(llava, bakllava 등)에서만 동작

        :param image_bytes:
        :param prompt:
        :param encoding:
        :return:
        """
        base64_image = base64.b64encode(image_bytes).decode(encoding)

        # HumanMessage에 이미지 + 텍스트 함께 넣기
        multimodal_msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
            ]
        )

        # 시스템 프롬프트 + 사용자 메시지
        messages = [
            SystemMessage(content=self.system_prompt or "당신은 시각 정보를 분석하는 AI입니다."),
            multimodal_msg
        ]
        return self.llm.invoke(messages)

    def invoke(self,message:str):
        pass
