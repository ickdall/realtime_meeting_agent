# 🧠 Real-Time Korean Meeting Analysis Agent

> 실시간 한국어 회의 발화를 요약하고, 질문 응답 및 관련 정보를 자동으로 분석하는 인공지능 기반 회의 분석 시스템입니다.

---

## 🎯 프로젝트 개요

현재 회의 분석 도구는 대부분 **영어 기반**이거나 **실시간 처리를 지원하지 않으며**,  
**한국어 특화 요약/질의응답 기능이 부재**합니다.

본 프로젝트는 다음을 목표로 합니다:

- ✅ **실시간 회의 발화 분석** (STT + LLM 요약)
- ✅ **질의응답 및 연관 정보 검색**
- ✅ **한국어 지원 특화 에이전트**
- ✅ **프론트 대시보드로 시각화 및 실시간 피드백**

---

## 🧩 주요 기능

| 기능 | 설명 |
|------|------|
| 🗣️ **음성 인식 (STT)** | CLOVA Speech-to-Text API를 통해 실시간 음성 → 텍스트 변환 |
| 🧠 **에이전트 분석** | LangChain 기반 GPT 모델로 요약, 질문 응답, 검색 수행 |
| 📡 **WebSocket 대시보드** | 프론트에서 실시간 텍스트 및 결과 스트리밍 시각화 |
| 📁 **데이터 저장** | 회의 텍스트, 요약, Q&A 결과를 로컬에 자동 저장 |

---

## ⚙️ 시스템 구성
 사용자 음성 ──▶ STT (Clova) ──▶ 텍스트 ──▶ LLM 분석 (GPT)

                           ▼
                    실시간 시각화 (WebSocket)

---
## 📌 기술 스택
FastAPI / Uvicorn – 백엔드 서버

Clova Speech-to-Text API – 한국어 음성 인식

LangChain + OpenAI GPT – 텍스트 요약, QA, 검색

WebSocket – 실시간 통신

React (JSX) – 프론트 대시보드

---

## 🙋‍♀️ 사용 예시
회의 중 실시간 요약 및 주요 질의응답

전화 상담 자동 분석 및 정리

한국어 컨퍼런스 실시간 기록 시스템

                    
