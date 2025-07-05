from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import json
import asyncio
from typing import List
from pydantic import BaseModel
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RealtimeText(BaseModel):
    text: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.recent_texts: List[str] = []  # 최근 텍스트 저장
        self.max_recent_texts = 5  # 최근 5개 발화만 유지
        self.current_sentence = ""  # 현재 진행 중인 문장
        self.accumulated_text = ""  # 누적 텍스트
        # LLM 결과를 저장할 딕셔너리
        self.last_llm_results = {
            "summaries": [],
            "qa_responses": [],
            "search_results": []
        }
        self.sentence_end_markers = ['.', '!', '?', '\n']  # 문장 종료 표시
        self.broadcast_lock = asyncio.Lock()  # 브로드캐스트 동기화를 위한 락

        # 로컬 파일 경로 설정
        self.base_path = r"C:\Users\user\Downloads\langchain-master\langchain-master\04_memory\text"
        self.file_paths = {
            'realtime': os.path.join(self.base_path, 'realtime'),
            'summary': os.path.join(self.base_path, 'summary'),
            'qa': os.path.join(self.base_path, 'qa'),
            'search': os.path.join(self.base_path, 'search')
        }
        for p in self.file_paths.values():
            os.makedirs(p, exist_ok=True)

    def is_sentence_end(self, text: str) -> bool:
        """문장 종료 여부 확인"""
        return any(marker in text for marker in self.sentence_end_markers)

    async def connect(self, websocket: WebSocket):
        """웹소켓 연결 수립"""
        await websocket.accept()
        self.active_connections.append(websocket)
        await self.send_initial_data(websocket)

    def disconnect(self, websocket: WebSocket):
        """웹소켓 연결 종료"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    def add_recent_text(self, text: str):
        """텍스트 추가 및 문장 관리"""
        self.current_sentence = (self.current_sentence + " " + text).strip()
        
        # 문장이 완성되었는지 확인
        if self.is_sentence_end(text):
            self.accumulated_text = (self.accumulated_text + " " + self.current_sentence).strip()
            self.recent_texts.append(self.current_sentence)
            if len(self.recent_texts) > self.max_recent_texts:
                self.recent_texts.pop(0)
            self.current_sentence = ""
            return True
        return False

    async def send_initial_data(self, websocket: WebSocket):
        """초기 연결 시 현재 상태 전송"""
        try:
            await websocket.send_json({
                "type": "initial_data",
                "current_sentence": self.current_sentence,
                "accumulated_text": self.accumulated_text,
                "llm_results": self.last_llm_results
            })
        except Exception as e:
            logger.error(f"Error sending initial data: {str(e)}")
            self.disconnect(websocket)

    async def broadcast(self, data: dict):
        """Broadcast updates to all WebSocket clients."""
        async with self.broadcast_lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    broadcast_data = {
                        **data,
                        "current_sentence": self.current_sentence,
                        "accumulated_text": self.accumulated_text,
                        "llm_results": self.last_llm_results
                    }
                    await connection.send_json(broadcast_data)
                except Exception as e:
                    logger.error(f"Broadcasting error: {str(e)}")
                    disconnected.append(connection)

            # 연결이 끊긴 클라이언트 제거
            for conn in disconnected:
                self.disconnect(conn)

    def save_text_to_local(self, text: str, file_type: str = 'realtime'):
        """텍스트를 로컬 파일에 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{file_type}_{timestamp}.txt"
            filepath = os.path.join(self.file_paths[file_type], filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)

            logger.info(f"Text saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save text: {str(e)}")
            return None

manager = ConnectionManager()

@app.post("/realtime-text")
async def handle_realtime_text(text_data: RealtimeText):
    """실시간 텍스트 처리"""
    try:
        # 텍스트 저장 및 브로드캐스트
        text_added = manager.add_recent_text(text_data.text)
        await manager.broadcast({
            "type": "text_update",
            "text": text_data.text,
        })

        # 텍스트 로컬 저장
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_text = f"[{timestamp}] {text_data.text}\n"
        manager.save_text_to_local(formatted_text)

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing realtime text: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/llm-results")
async def handle_llm_results(request: Request):
    """LLM 결과 처리"""
    try:
        results = await request.json()

        # 최신 LLM 결과를 manager 객체에 반영
        manager.last_llm_results["summaries"].extend(results.get("summaries", []))
        manager.last_llm_results["qa_responses"].extend(results.get("qa_responses", []))
        manager.last_llm_results["search_results"].extend(results.get("search_results", []))

        # 결과 로컬 저장
        for result_type in ['summary', 'qa', 'search']:
            if results.get(f"{result_type}s"):
                for content in results[f"{result_type}s"]:
                    manager.save_text_to_local(content, result_type)

        # 모두에게 브로드캐스트
        await manager.broadcast({
            "type": "llm_update",
            "results": results
        })

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing LLM results: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """상태 확인 엔드포인트"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)