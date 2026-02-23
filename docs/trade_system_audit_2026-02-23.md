# Trade System Audit (2026-02-23)

## 범위 / 가정
- 서버 스타트업이 완료되어 `league.current_date` 등 필수 런타임 전제는 만족한다고 가정.
- 정상적인 게임 흐름(거래 요청 → `parse_deal`/`validate_deal` → 실행)을 기준으로, 트레이드 SSOT 위반 여부를 점검.

## 검증 결론
- **치명적 SSOT 위반 1건 확인**.
- 위반 내용: `parse_deal`이 `teams`에 없는 `legs` 키를 **에러 없이 조용히 삭제**함.

## 확인된 문제 (재현 3회 완료)

### 1) `parse_deal`의 조용한 legs 삭제 (Silent Drop)
`parse_deal`은 입력 `legs` 전체를 정규화한 뒤, `for team_id in teams` 루프에서 팀 목록에 포함된 키만 다시 채택한다.
즉, payload에 추가로 포함된 팀 legs(예: `CHI`)는 예외 없이 버려진다.

### 비개발자용 쉬운 설명
이 문제를 아주 쉽게 말하면, **"신청서에 적은 내용 일부를 시스템이 몰래 지우고 처리하는 상황"**입니다.

- 트레이드 요청에는 보통
  - `teams`: 거래 당사자 팀 목록
  - `legs`: 각 팀이 내보내는 자산 목록
  이 함께 들어갑니다.
- 그런데 지금은 `teams`에 없는 팀이 `legs`에 들어 있어도, 시스템이 "오류"라고 알려주지 않습니다.
- 대신 그 팀의 내용을 조용히 삭제한 뒤, "정상 요청"처럼 다음 단계로 넘깁니다.

#### 일상 비유
- 택배 접수서에
  - 받는 사람: A, B
  - 배송 물품: A 박스, B 박스, C 박스
  를 적었는데,
- 창구에서 "C 박스는 대상에 없으니 접수 불가"라고 말해야 정상입니다.
- 하지만 현재 동작은 **아무 말 없이 C 박스를 빼고 접수**해버리는 것과 같습니다.

#### 왜 문제인가?
- 요청한 사람 입장: "분명 C도 넣었는데 왜 결과에 없지?"
- 운영/CS 입장: 에러가 없으니 어디서 잘못됐는지 추적이 매우 어려움.
- 신뢰성 관점: "입력한 내용"과 "실제로 검증/처리된 내용"이 달라져서 시스템 신뢰가 떨어짐.

#### 기대되는 올바른 동작
- `teams`에 없는 팀이 `legs`에 있으면,
- 시스템이 즉시 "잘못된 요청"으로 거절하고,
- "어떤 팀 키가 잘못됐는지"를 명확한 에러로 알려줘야 합니다.

- 관련 코드:
  - `normalized_legs_raw`를 만들지만 (`legs_raw` 전체 보유),
  - 최종 `legs`는 `teams`에 있는 키만 채워 return.
- 결과:
  - 클라이언트 입력 payload와 서버가 실제로 검증/실행하는 deal 객체가 불일치 가능.
  - `TeamLegsRule`의 "legs == teams" 검증 이전에, payload가 이미 변형되어 검증이 우회되는 형태.

### 내부 시뮬레이션 3회
아래 동일 커맨드 1회 실행 안에서 3개 케이스를 순차 수행:

```bash
python - <<'PY'
from trades.models import parse_deal, serialize_deal
cases=[
 {'teams':['ATL','BOS'],'legs':{'ATL':[],'BOS':[],'CHI':[{'kind':'player','player_id':'P000001'}]}},
 {'teams':['LAL','MIA'],'legs':{'LAL':[{'kind':'pick','pick_id':'2026_LAL_R1'}],'MIA':[],'NYK':[{'kind':'pick','pick_id':'2027_NYK_R2'}]}},
 {'teams':['PHX','SAS'],'legs':{'PHX':[{'kind':'fixed_asset','asset_id':'CASH_1'}],'SAS':[],'DEN':[{'kind':'fixed_asset','asset_id':'CASH_2'}]}}
]
for i,p in enumerate(cases,1):
 d=parse_deal(p)
 s=serialize_deal(d)
 print('SIM',i,'input_legs',sorted(p['legs'].keys()),'parsed_legs',sorted(s['legs'].keys()),'dropped',sorted(set(p['legs'])-set(s['legs'])))
PY
```

관측 결과:
- SIM 1: `CHI` leg 삭제
- SIM 2: `NYK` leg 삭제
- SIM 3: `DEN` leg 삭제

즉, 우연/단일 입력 특이치가 아니라 **일관 동작**으로 확인됨.

## 영향도 평가
- SSOT 관점에서 **높음(High)**:
  - "입력 payload == 검증 대상" 원칙이 깨짐.
  - 잘못된 payload가 "거절"되지 않고 "변형 수용"되어 운영 측 디버깅 난이도 증가.
- 게임 흐름 관점:
  - 즉시 크래시보다는 "조용한 데이터 의미 손실" 타입의 결함.
  - 거래 생성기/외부 클라이언트/툴링에서 payload mismatch가 발생하면 원인 추적이 어려움.

## 권장 조치
- `parse_deal`에서 다음 검증 추가:
  - `extra_leg_teams = set(normalized_legs_raw.keys()) - set(teams)`가 비어있지 않으면 `DEAL_INVALIDATED`로 즉시 실패.
- 실패 payload에 `extra_leg_teams`를 넣어 문제를 명시적으로 노출.
