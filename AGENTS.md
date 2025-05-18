# AGENTS Guide

## 1. 언어 규칙
- **에이전트-사용자 대화**: 한국어  
- **코드, PR 제목·본문, 커밋 메시지, 주석**: 영어

---

## 2. 브랜치 & 커밋 규칙

| 항목        | 최소 지침                                  |
|-------------|-------------------------------------------|
| **브랜치**  | `feat/<topic>` 또는 `fix/<topic>`          |
| **커밋**    | `<type>: <short summary>` <br>예) `feat: add PID controller`<br>`type` 값은 `feat`, `fix`, `docs`, `refactor` 정도만 사용 |

---

## 3. 코드 품질 도구
- **Formatter**: **Black** (자동 코드 정렬)  
- **Linter**&nbsp;&nbsp;&nbsp;: **Ruff** (Python 스타일·버그 검사)  
  > Black 과 Ruff는 `pre-commit` 훅에 연결되어 있어 커밋 전에 자동 실행됩니다.


## 4. PR 체크리스트
제목: [lerobot] <영어 제목>

본문: 변경 요약(영어)
