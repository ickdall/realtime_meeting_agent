from openai import OpenAI
import time
import pyaudio
import grpc
import nest_pb2
import nest_pb2_grpc
import json
import threading
from threading import Lock
import wave
import io
from datetime import datetime
from queue import Queue
import os
import requests
import re
from bs4 import BeautifulSoup
import html
from typing import Dict, List
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableAssign
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
# from langchain_teddynote.messages import stream_response
from langchain_core.output_parsers import StrOutputParser
# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] ="API_KEY"
# 네이버 search api
naver_search_id = 'CLOVA_API_ID'
naver_search_pw = 'CLOVA_API_Password'


# CLOVA Speech 설정
CLIENT_SECRET = 'CLOVA_API_KEY'
INVOKE_URL ='CLOVA_API_URL'

summary_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


# 각각의 프롬프트 지정

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 회의나 발표의 주요 내용을 실시간으로 요약하는 AI 어시스턴트입니다. 
        목표는 중요한 세부 사항을 놓치지 않으면서 간결하고 유익한 요약을 제공하는 것입니다. 아래는 요약을 작성할 때 따를 수 있는 템플릿입니다. 모든 내용을 처리할 때 이 구조를 따르세요.
        
        실시간 문장에 다음과 같은 양식으로 사용자에게 정보를 제공합니다.
        *********
        회의/발표 기본 템플릿 (적용 가능한 경우)

        회의 안건/프레젠테이션 주제: # 문장에서 주제가 나오면 이 양식으로 출력합니다.
        날짜 및 시간: # 날짜를 알 수 있으면 그 다음으로 출력합니다.
        발표자/연설자 이름: # 알 수 있으면 실시합니다.
        
        각 섹션 요약 
        각각의 회의나 발표 섹션에 대해 실시간으로 내용을 기록하며, 관련된 주요 내용만을 요약하십시오.
        핵심적인 논의 내용이나 발표의 주요 포인트를 간결하게 요약합니다.
        각 발언자의 핵심 아이디어 및 주장
        각 구체적인 내용을 섹션별로
        참석자들의 질문 및 발표자/연설자의 답변 요약(해당시)
        하나의 안건, 주장에 대한 요약 
        새로운 아이디어, 주제시 위의 섹션요약으로 다시 시작
        논의된 주제의 결론 또는 후속 조치
        향후 계획 또는 진행 사항
        *********
        실시간 음성을 입력하고 있기 때문에, 프로젝트,회의와 관련된 문장에 대해서만 형식에 맞게 제공하면 됩니다.
        회의와 관련없는 **무조건 무응답해야합니다**.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 사용자의 요청에 따라 gpt 학습된 데이터를 통해 질의응답을 하는 에이전트입니다.
        없는 정보를 만들어내서는 안됩니다.
        사용자의 요청에 따라 성심성의껏 중요한 내용만 요약해야합니다.

        """),
        ("human", "{input}"),
    ]
)

# 
confernce_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        당신은 사용자의 요청에 따라 gpt 학습된 데이터를 통해 질의응답을 하는 에이전트입니다.
        없는 정보를 만들어내서는 안됩니다.
        사용자의 요청에 따라 성심성의껏 작성해야합니다

        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

controll_prompt=ChatPromptTemplate.from_messages(
   [ 
    ("system",
     """"
     당신은 다른 에이전트들을 관리하는 Controll Agent입니다.
     실시간으로 문장들이 완성돼서 입력됩니다. 그렇기에 문장을 파악해서 다른 에이전트를 작동시킬지 정확히 파악해야합니다.

     당신이 관리해야할 에이전트는 총 2가지 에이전트입니다.

     **qa 에이전트**: GPT 내부에 있는 데이터를 가지고 사용자의 질문에 응답할때 사용하는 에이전트 입니다.
     무조건 검색이라고 seach 에이전트를 호출하는 것이 아닌, gpt가 학습한 데이터로 충분히 답변이 가능하거나 gpt와의 질의응답이 필요하거나, 회의에 대한 요약, 의견 조율이 필요할때 이 에이전트를 활성화 시킵니다.
     수신 받은 문장이 진행중인 회의 내용일 경우 호출시키면 안됩니다. 'gpt야 물어볼게'와 같이 정보를 요구하면서 외부검색이 아닌경우만 실행시킵니다.

     **search 에이전트**: gpt내부에 있는 데이터로는 대답할 수 없는 외부 검색이 필요할때 사용하는 에이전트 입니다. 
     사용자의 대화에서  최근 동향이나 이슈에 대한 정보가 필요하거나 ,어떤 물건이 부족하다거나, 망가졌을때 당신은 능동적으로 이 에이전트를 활성화해서 사용자에게 편의를 줘야합니다.
    

     
     search 에이전트를 활성화 시킬땐 search, qa 에이전트를 활성화 시킬땐 qa, 아무런 호출이 필요없을땐 N으로 응답합니다.
   
     """),
    ("human","{user_input}")
    ]
    )

search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 능동적으로 사용자의 대답을 보고 적재적소에 네이버 검색 API를 사용하여 정보를 제공하는 에이전트입니다.
            문맥을 보고 검색을 통한 정보제공이 도움이 되면 검색하고, 들어 외부 검색이 필요없다면 함수를 실행시키지 않고 종료합니다.
            예를 들어서 어떤 물건이 부족하다거나, 망가졌을때 당신은 능동적으로 검색해서 정보를 제공해줄 수 있습니다.
            
            search="query,obj"텍스트 형태로 구성됩니다. 두개의 인자만 무조건 아래 로직을 통해 담겨져야 합니다
            - query: 요청에서 검색에 최적화된 키워드로 변환합니다
            - obj: 검색 유형을 지정 (선택적)
            - 'news': 뉴스 검색 (특정 주제의 최신 뉴스나 기사를 찾을 때 사용)
            - 'local': 지역 정보 검색 (맛집, 가게, 장소 등을 찾을 때 사용)
            - 'shop': 상품 검색 (제품 정보나 가격을 찾을 때 사용)
            
            질문자의 요청: "컴퓨터좀 사려하는데 게임하기에 좋은 컴퓨터좀 알아봐줘"  
            변환: 함수 변수 입력 예시: "게이밍 컴퓨터,shop"입니다.
            검색 결과를 사용자에게 전달하기 좋은 형태로 제공합니다.
            # 검색한 모든 정보에 아래와 같은 형식으로 답변해야합니다.
            
            # *뉴스 검색에 대한 답변형식

            # 안녕하세요! 삼성 주가 동향에 대한 기사를 요약해드리겠습니다.
            # 3개의 기사를 분석하겠습니다.

            # **신문 제목: 젠슨 황 한마디에...삼성SK 주가 들썩**
            # **분석 내용:** 이 기사는 젠슨 황이 엔비디아의 그래픽 처리장치(GPU) 신제품에 삼성전자의 메모리칩이 들어가지 않는다고 한 발언을 정정하면서 삼성전자와 SKC의 주가 변동에 대해 다루고 있습니다. 젠슨 황이 최근 삼성전자의 메모리칩이 사용될 것이라고 발언한 후, 이 발언을 정정하면서 삼성전자와 SKC의 주가가 상승하거나 하락한 것을 다루고 있습니다.
            # **링크:(http://www.edaily.co.kr/news/newspath.asp?newsid=03962246642036080)

            # 기사들을 분석해보면 단기적으로는 젠슨 황의 발언이 삼성전자의 주가에 부정적인 영향을 미쳤을 수 있으나, 발언이 정정되면서 주가가 회복되었을 가능성이 큽니다.
            # 장기적으로 볼 때, 삼성전자의 반도체 사업은 여전히 강력한 경쟁력을 가지고 있으며, 엔비디아와의 협력은 중요한 요소로 작용할 수 있습니다. 따라서, 이번 사건은 주가에 큰 영향을 미친 사건이라기보다는 일시적인 변동성을 나타내는 사례라고 할 수 있습니다.
            # -----------------------------------
            # *obj=shop 제품 검색 답변형식
            
            # 이러한 query 상품을 추천드립니다

            # 제품명: 
            # 가격:
            # 판매처: 
            # 브랜드: 
            # [상세 정보를 확인하세요]:링크         
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)



class SummaryAgent():
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model="gpt-4",  # 모델명 수정
            temperature=0,
        )
        self.memory = summary_memory
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                |itemgetter(self.memory.memory_key)
            )
            | summary_prompt
            | self.model_structure
            | StrOutputParser()
        )
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.sentence = ""
        self.count = 0
        self.last_summary = None
        self.is_running = True
        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()

    def process_tasks(self):
        if not self.task_queue.empty():
            self.sentence += self.task_queue.get()
            self.count += 1
            if self.count == 1:
                self.summary_process_tasks()

    def summary_process_tasks(self):
        answer = self.chain.invoke({"input": self.sentence})
        print('요약 에이전트 응답입니다.')
        print(answer)
        self.memory.save_context(inputs={"human": self.sentence}, outputs={"ai": answer})
        self.last_summary = answer  # 결과 저장
        self.sentence = ""
        self.count = 0

        
class QAAgent():
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model="gpt-4",  # 모델명 수정
            temperature=0,
        )
        self.memory = summary_memory
        self.chain = RunnablePassthrough()|qa_prompt|self.model_structure
        self.memory_chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                |itemgetter(self.memory.memory_key)
            )
            | summary_prompt
            | self.model_structure
            | StrOutputParser()
        )
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.sentence = ""
        self.count = 0
        self.last_response = None
        self.is_running = True
        self.thread = threading.Thread(target=self.process_tasks)
        self.thread.daemon = True
        self.thread.start()

    def process_tasks(self):
        if not self.task_queue.empty():
            self.sentence += self.task_queue.get()
            self.count += 1
            if self.count == 1:
                self.qa_process_tasks()

    def qa_process_tasks(self):
        answer = self.chain.invoke(self.sentence)
        print('qa 에이전트의 응답입니다')
        print(answer.content)
        self.last_response = answer.content  # 결과 저장
        self.sentence = ""
        self.count = 0

class SearchAgent():
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model="gpt-4",  # 모델명 수정 
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.prompt = search_prompt
        self.task_que = Queue()
        self.result_que = Queue()
        self.last_result = None  # 결과 저장을 위한 변수 추가

    @tool
    def Naver_Search(search: str) -> str:
        """
        Naver 검색 기능을 수행하는 함수입니다.
        주어진 검색어를 사용하여 네이버에서 관련된 결과를 반환합니다.

        Args:
            search (str): "query,obj" 형태로 받아옵니다.

        Returns:
            str: 검색 결과로 반환된 텍스트 (예: 뉴스, 블로그, 등)
        """
        query, obj = search.split(",")

        try:
            sort = 'sim'
            encText = requests.utils.quote(query)

            if obj == 'news':
                display = 3
            else:
                display = 5

            url = f"https://openapi.naver.com/v1/search/{obj}?query={encText}&display={display}&sort={sort}"
            
            if obj == 'shop':
                url += '&exclude=used:cbshop'

            headers = {
                "X-Naver-Client-Id": naver_search_id,
                "X-Naver-Client-Secret": naver_search_pw
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                info = response.json()
            else:
                return f"Error Code: {response.status_code} - 검색 결과를 찾지 못했습니다."

            def clean_unicode_characters(text):
                text = re.sub(r"[\u2018\u2019\u201C\u201D\u2022\u2026\u2013\u2014\u00B7\uFF0C\uFF1A\u3002]", "", text)
                return text
            
            def naver_crawl_news(a):
                cnt = 0
                news_text = ""
                if a:
                    for item in a['items']:
                        if cnt < 4:
                            response = requests.get(item['originallink'])
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, 'html.parser')
                                title = html.unescape(item['title'])
                                title = re.sub(r"[\'\"\,]+|<b>|</b>|", "", title)
                                title = clean_unicode_characters(title)
                                description = html.unescape(item['description'])
                                description = re.sub(r"[\'\"\,]+|<b>|</b>|", "", description)
                                description = clean_unicode_characters(description)
                                context = re.sub(r"[\n\r\'\"\,]+|<b>|</b>|\s{2,}", "", soup.text)
                                context = clean_unicode_characters(context)
                                link = item['originallink']
                                matches = list(re.finditer(re.escape(title), context, re.IGNORECASE))

                                if matches:
                                    last_match = matches[-1]
                                    end_index = last_match.end()
                                    extract_context = context[end_index:]

                                    news_text += f'신문 제목: {title}\n'
                                    news_text += f'요약문: {description}\n'
                                    news_text += f'본문 일부: {extract_context[150:450]}\n'
                                    news_text += f'신문기사 링크: {link}\n\n'
                                cnt += 1
                return news_text

            def naver_product_info(a):
                product_info = ""
                if a:
                    for item in a['items']:
                        product_info += f"""
                            {item['image']}
                            제품명: {re.sub(r"</b>|<b>","",html.unescape(item['title']))}
                            가격: {item['lprice']}원
                            판매처: {item['mallName']}
                            브랜드: {item['brand']}
                            [상세 정보를 확인하세요] {item['link']}
                            """
                return product_info

            def naver_food_info(a):
                food_info = ""
                if a:
                    for item in a['items']:
                        food_info += f"""
                            가게명: {re.sub(r"</b>|<b>","",html.unescape(item['title']))}
                            음식 카테고리: {item['category']}
                            위치: {item['address']}
                            """
                return food_info
            
            if obj == 'news':
                return naver_crawl_news(info)
            elif obj == 'shop':
                return naver_product_info(info)
            elif obj == 'local':
                return naver_food_info(info)
        except Exception as e:
            print(f"Naver Search error: {str(e)}")
            return None

    def process_tasks(self):
        """검색 작업 처리 및 결과 저장"""
        if not self.task_que.empty():
            query = self.task_que.get()
            tools = [self.Naver_Search]
            
            # Agent 생성
            agent = create_tool_calling_agent(self.model_structure, tools, self.prompt)

            # AgentExecutor 생성
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=10,
                max_execution_time=10,
                handle_parsing_errors=True,
            )
            
            # AgentExecutor 실행
            try:
                result = agent_executor.invoke({"input": query})
                print(result)
                # 검색 결과 저장
                if result and 'output' in result:
                    self.last_result = result['output']
                    return result['output']
            except Exception as e:
                print(f"Error in search execution: {str(e)}")
                return None


class MultiAgentProcessor:
    def __init__(self):
        self.model_structure = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0,
        )
        self.summary_agent = SummaryAgent()
        self.qa_agent = QAAgent()
        self.search_agent = SearchAgent()
        
        self.chain = RunnablePassthrough()|controll_prompt|self.model_structure
        self.text_queue = Queue()
        self.last_texts = ""
        self.sentence_que = Queue()
        self.is_running = True

        # 로컬 파일 경로 설정
        self.base_path = r"./data/realtime"
        self.paths = {
            'realtime': os.path.join(self.base_path, 'realtime'),
            'summary': os.path.join(self.base_path, 'summary'),
            'qa': os.path.join(self.base_path, 'qa'),
            'search': os.path.join(self.base_path, 'search')
        }
        
        # 디렉토리 생성
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

        self.processing_thread = threading.Thread(target=self._process_text_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()


    def save_to_file(self, content, file_type):
        """파일 저장 및 서버 전송 함수"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{file_type}_{timestamp}.txt"
            filepath = os.path.join(self.paths[file_type], filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"File saved: {filepath}")

            # 서버로 결과 전송
            result_data = {
                "summaries": [content] if file_type == 'summary' else [],
                "qa_responses": [content] if file_type == 'qa' else [],
                "search_results": [content] if file_type == 'search' else [],
                "type": file_type,
                "timestamp": timestamp
            }

            try:
                requests.post(
                    'http://localhost:8000/llm-results',
                    json=result_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=1
                )
            except Exception as e:
                print(f"Failed to send results to server: {str(e)}")

            return filepath
        except Exception as e:
            print(f"Failed to save file ({file_type}): {str(e)}")
            return None
        
    def process_text(self, text):
        """텍스트 처리 로직"""
        self.text_queue.put(text)
        
        # 실시간 텍스트 프론트엔드로 전송
        try:
            requests.post(
                'http://localhost:8000/realtime-text',
                json={'text': text},
                headers={'Content-Type': 'application/json'},
                timeout=0.5
            )
        except Exception as e:
            print(f"Failed to send realtime text: {str(e)}")
        
        # 실시간 텍스트 파일 저장
        filepath = os.path.join(self.paths['realtime'], f"realtime_{datetime.now().strftime('%Y%m%d')}.txt")
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')
                f.write(f"[{timestamp}] {text}\n")
        except Exception as e:
            print(f"Error saving realtime text: {str(e)}")

    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)

    
    def _process_text_queue(self):
        """텍스트 처리 큐 프로세스"""
        while self.is_running:
            try:
                if not self.text_queue.empty():
                    new_text = self.text_queue.get()
                    self.last_texts += new_text

                    sentences = re.split(r"(?<=[.!?])\s*", self.last_texts.strip())
                    
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                            self.sentence_que.put(sentence.strip())
                        self.last_texts = sentences[-1]
                    
                    if not self.sentence_que.empty():
                        sentence_to_process = self.sentence_que.get()
                        self.controll_task(sentence_to_process)
                       
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in text processing: {str(e)}")

    def controll_task(self, sentence_to_process):
        """컨트롤 에이전트 업무 실행"""
        try:
            # 요약 처리
            try:
                self.summary_agent.task_queue.put(sentence_to_process)
                self.summary_agent.process_tasks()
                
                if self.summary_agent.last_summary:
                    self.save_to_file(self.summary_agent.last_summary, 'summary')
                
            except Exception as e:
                print(f"Summary agent error: {str(e)}")
            
            # 컨트롤러 응답 처리
            answer = self.chain.invoke(sentence_to_process)
            print(f"controll agent 응답 결과: {answer.content}")
            
            if answer.content == "search":
                self.search_agent.task_que.put(sentence_to_process)
                self.search_agent.process_tasks()
                if self.search_agent.last_result:
                    self.save_to_file(self.search_agent.last_result, 'search')
                    
            elif answer.content == "qa":
                self.qa_agent.task_queue.put(sentence_to_process)
                self.qa_agent.process_tasks()
                if self.qa_agent.last_response:
                    self.save_to_file(self.qa_agent.last_response, 'qa')
                    
        except Exception as e:
            print(f"Error in control task: {str(e)}")

class CombinedSpeechRecognizer:
    def __init__(self):
        self.running = True
        self.audio_frames = []
        self.recording_start_time = time.time()
        self.current_interval = 1
        self.interval_duration = 10
        self.termination_phrases = [
            "회의를 종료하겠습니다",
            "마치겠습니다",
            "회의를 마치겠습니다",
            "마칠게",
            "종료"
        ]
        self.text_processor = MultiAgentProcessor()
        
        self.base_path = r"./data/realtime"
        self.paths = {
            'realtime': os.path.join(self.base_path, 'realtime'),
            'audio': os.path.join(self.base_path, 'audio')
        }
        
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
        self.current_realtime_file = os.path.join(
            self.paths['realtime'], 
            f"realtime_{datetime.now().strftime('%Y%m%d')}.txt"
        )

            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 스트림 콜백"""
        self.audio_frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def save_text_to_local(self, text):
        """실시간 인식 텍스트를 파일에 저장"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_text = f"[{timestamp}] {text}\n"
            with open(self.current_realtime_file, 'a', encoding='utf-8') as f:
                f.write(formatted_text)
            # print(f"텍스트 저장 완료: {self.current_realtime_file}")
            return self.current_realtime_file
        except Exception as e:
            # print(f"텍스트 저장 실패: {str(e)}")
            return None

    def save_audio(self):
        """현재까지의 오디오 프레임을 파일로 저장"""
        if not self.audio_frames:
            return None

        frames_to_save = self.audio_frames.copy()
        self.audio_frames = []

        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(self.paths['audio'], f"audio_{current_timestamp}.wav")
        
        try:
            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames_to_save))
            print(f"오디오 파일 저장: {file_name}")
            return file_name
        except Exception as e:
            print(f"오디오 파일 저장 실패: {str(e)}")
            return None

    def periodic_audio_save(self):
        """10초마다 오디오 저장"""
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.recording_start_time
            
            if elapsed_time >= (self.current_interval * self.interval_duration):
                # print(f"\n{self.current_interval * self.interval_duration}초 녹음 완료")
                self.save_audio()
                self.current_interval += 1
            time.sleep(1)

    def check_termination(self, text):
        """종료 문구 확인"""
        return any(phrase in text.lower() for phrase in self.termination_phrases)

    def generate_requests(self):
        """음성 인식 요청 생성"""
        yield nest_pb2.NestRequest(
            type=nest_pb2.RequestType.CONFIG,
            config=nest_pb2.NestConfig(
                config=json.dumps({
                    "transcription": {"language": "ko"},
                    "semanticEpd": {
                        "useWordEpd": "true",
                        "syllableThreshold": "4",
                        "durationThreshold": "3000",
                        "gapThreshold": "1000"
                    }
                })
            )
        )
                
        p = pyaudio.PyAudio()
        print("\n=== 사용 가능한 마이크 목록 ===")
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"번호 {i}: {device_info['name']}")
                    print(f"   채널 수: {device_info['maxInputChannels']}")
                    print(f"   샘플레이트: {int(device_info['defaultSampleRate'])}Hz")
                    print("---")
            except Exception:
                continue

        try:
            default_device = p.get_default_input_device_info()
            print(f"\n기본 마이크: {default_device['name']} (번호: {default_device['index']})")
        except:
            print("\n기본 마이크를 찾을 수 없습니다.")

        while True:
            try:
                device_index = input('\n사용할 마이크 번호를 입력하세요 (기본값: Enter): ').strip()
                if not device_index:
                    device_index = p.get_default_input_device_info()['index']
                    break
                device_index = int(device_index)
                if 0 <= device_index < p.get_device_count():
                    device_info = p.get_device_info_by_index(device_index)
                    if device_info['maxInputChannels'] > 0:
                        break
                print("올바르지 않은 마이크 번호입니다. 다시 시도해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=3200
        )
        
        print("\n음성 인식 시작...")

        try:
            while self.running:
                chunk = stream.read(3200, exception_on_overflow=False)
                self.audio_frames.append(chunk)
                yield nest_pb2.NestRequest(
                    type=nest_pb2.RequestType.DATA,
                    data=nest_pb2.NestData(
                        chunk=chunk,
                        extra_contents=json.dumps({
                            "seqId": len(self.audio_frames),
                            "epFlag": False
                        })
                    )
                )
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def process_responses(self, responses, metadata):
        """실시간 인식 응답 처리"""
        for response in responses:
            if not self.running:
                break

            try:
                if hasattr(response, 'result'):
                    text = response.result.hypotheses[0].text if response.result.hypotheses else ""
                else:
                    response_str = (
                        response.contents if isinstance(response.contents, str) 
                        else response.contents.decode('utf-8')
                    )
                    response_json = json.loads(response_str)
                    text = response_json.get('text', '')
                    if not text and 'transcription' in response_json:
                        text = response_json['transcription'].get('text', '')

                if text:
                    print(f"실시간 인식 텍스트: {text}")

                    # 실시간 텍스트 프론트엔드로 전송
                    def send_text():
                        try:
                            requests.post(
                                'http://localhost:8000/realtime-text',
                                json={'text': text},
                                headers={'Content-Type': 'application/json'},
                                timeout=0.5
                            )
                        except Exception as e:
                            print(f"실시간 텍스트 전송 실패: {str(e)}")

                    threading.Thread(target=send_text, daemon=True).start()

                    # LLM 처리 (텍스트 저장 포함)
                    threading.Thread(
                        target=self.text_processor.process_text,
                        args=(text,),
                        daemon=True
                    ).start()

                    if self.check_termination(text):
                        print("\n종료 문구 감지 → 프로그램 종료")
                        self.running = False
                        break

            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {str(e)}")
            except Exception as e:
                print(f"응답 처리 중 오류: {str(e)}")
                continue

    def cleanup(self):
        """프로그램 종료 시 정리"""
        self.running = False
        if hasattr(self.text_processor, 'cleanup'):
            self.text_processor.cleanup()

    def run(self):
        """프로그램 실행"""
        save_thread = threading.Thread(target=self.periodic_audio_save)
        save_thread.start()
        
        channel = grpc.secure_channel(
            "clovaspeech-gw.ncloud.com:50051",
            grpc.ssl_channel_credentials(),
            options=[
                ('grpc.max_receive_message_length', 1024 * 1024 * 10),
                ('grpc.max_send_message_length', 1024 * 1024 * 10),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000)
            ]
        )
        
        stub = nest_pb2_grpc.NestServiceStub(channel)
        metadata = (("authorization", f"Bearer {CLIENT_SECRET}"),)
        
        try:
            print("=== 음성 인식을 시작합니다 ===")
            print("10초마다 자동으로 오디오가 저장됩니다.")
            print("종료하려면 '회의를 종료하겠습니다' 등의 멘트를 말씀해주세요.\n")
            
            responses = stub.recognize(self.generate_requests(), metadata=metadata)
            self.process_responses(responses, metadata)
        except grpc.RpcError as e:
            print(f"gRPC 오류 발생: {e.details() if hasattr(e, 'details') else str(e)}")
        except Exception as e:
            print(f"예상치 못한 오류 발생: {str(e)}")
        finally:
            self.cleanup()
            channel.close()
            save_thread.join(timeout=2)
            print("\n프로그램이 종료되었습니다.")

if __name__ == "__main__":
    recognizer = CombinedSpeechRecognizer()
    recognizer.run()
