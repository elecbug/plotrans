"""Scoped Korean terms and table labels for scientific papers."""

from __future__ import annotations

import re

_KOREAN = {'korean', 'ko', 'ko-kr', '한국어'}
TABLE_HEADERS = {
    'component': '구성 요소', 'specification': '사양', 'metric': '지표',
    'parameter': '매개변수', 'value': '값', 'description': '설명',
    'result': '결과', 'name': '이름', 'type': '유형', 'category': '범주',
}


def table_header_translation(source: str, language: str) -> str | None:
    """Disambiguate ordinary table labels, e.g. metric versus the metric system."""
    return TABLE_HEADERS.get(source.strip().casefold()) if language.strip().casefold() in _KOREAN else None


_NETWORK = re.compile(r'\b(?:peer[ -]to[ -]peer|P2P|GossipSub|blockchain|bitcoin|cryptocurrency|network(?:ing)?|mesh|routing)\b', re.I)
_GRAPH = re.compile(r'\b(?:graphs?|topolog\w*|mesh|GossipSub|nodes?|vertices|vertex|edges?)\b', re.I)
_OTHER_DEGREE = re.compile(
    r'\b(?:celsius|fahrenheit|kelvin|temperature|angles?|rotation|polynomial|bachelor|master|doctoral|academic degree)\b'
    r'|\bdegrees?\s+of\s+(?!(?:(?:a|an|the|each|every|this|that)\s+)?(?:nodes?|vertices|vertex|graphs?|peers?)\b)', re.I)
_CUSTOMERS = re.compile(r'\b(?:customers?|customer retention|subscriptions?)\b', re.I)
_SHIPPING = re.compile(r'\b(?:goods|shipping|shipments?|couriers?)\b', re.I)
_RATE_UNITS = re.compile(r'(?<![A-Za-z])(?:messages?|packets?|bits?|bytes?|transactions?)\s*(?:/\s*s(?:ec(?:ond)?)?|per\s+(?:second|minute|hour))\b|(?<![A-Za-z])[kmgt]?(?:bps|b/s|bit/s|byte/s)\b', re.I)

# Longer phrases take precedence over words inside those phrases. Only terms
# present outside protected mathematical content are sent to the model.
_RULES = (
    (r'\bNational Research Foundation of Korea\b', '한국연구재단', 'entity'),
    (r'\bdegree[ -]control parameters?\b', '차수 제어 매개변수', 'graph'),
    (r'\bdegree[ -]control settings?\b', '차수 제어 설정', 'graph'),
    (r'\bdegree[ -]control\b', '차수 제어', 'graph'),
    (r'\bdegree[ -]distributions?\b', '차수 분포', 'graph'),
    (r'\bdegree[ -]assortativity\b', '차수 동류성', 'graph'),
    (r'\bdegree[ -]variance\b', '차수 분산', 'graph'),
    (r'\beager[ -]mesh degree\b', '즉시 전파 메시 차수', 'graph'),
    (r'\bgossip degree\b', '가십 차수', 'graph'),
    (r'\bmesh degree\b', '메시 차수', 'graph'),
    (r'\b(?:average|mean) degree\b', '평균 차수', 'graph'),
    (r'\btarget degree\b', '목표 차수', 'graph'),
    (r'\bnode degree\b', '노드 차수', 'graph'),
    (r'\bclustering coefficients?\b', '군집 계수', 'graph'),
    (r'\baverage shortest path length\b', '평균 최단 경로 길이', 'graph'),
    (r'\bGS Delivery\b', 'GS 전달 성공률', 'delivery'),
    (r'\b(?:message )?delivery ratio\b', '메시지 전달 성공률', 'delivery'),
    (r'\b(?:message )?delivery rate\b', '메시지 전달 성공률', 'delivery-rate'),
    (r'\bchurn rates?\b', '노드 변동률', 'churn'),
    (r'\bchurn\b', '노드 변동', 'churn'),
    (r'\beager[ -]push\b', '즉시 전파', 'network'),
    (r'\blazy[ -]pull\b', '지연 요청', 'network'),
    (r'\bmesh neighbou?rs\b', '메시 이웃', 'network'),
    (r'\baverage hop[ -]distance\b', '평균 홉 거리', 'network'),
    (r'\bminers?\b', '채굴 노드', 'mining'),
    (r'\bphysical servers?\b', '물리 서버', 'cluster-roles'),
    (r'\bmanagers?\b', '관리 노드', 'cluster-roles'),
    (r'\bworkers?\b', '작업 노드', 'cluster-roles'),
    (r'\bassortativity\b', '동류성', 'graph'),
    (r'\bmodularity\b', '모듈성', 'graph'),
    (r'\bdegrees?\b', '차수', 'degree'),
    (r'\bdelivery\b', '전달', 'delivery'),
)


def translation_terms(source: str, context: str, language: str) -> tuple[tuple[str, str], ...]:
    """Select terminology, without passing paper titles or abstract facts."""
    if language.strip().casefold() not in _KOREAN:
        return ()
    topic = source + '\n' + context
    network = bool(_NETWORK.search(topic))
    graph = bool(_GRAPH.search(topic))
    enabled = {
        'entity': True,
        'network': network,
        'graph': graph,
        'degree': graph and not _OTHER_DEGREE.search(source),
        'churn': network and not _CUSTOMERS.search(topic),
        'delivery': network and not _SHIPPING.search(topic),
        'delivery-rate': network and not _SHIPPING.search(topic) and not _RATE_UNITS.search(source),
        'mining': bool(re.search(r'\b(?:cryptocurrency|blockchain|bitcoin|peer[ -]to[ -]peer|P2P|gossip(?:sub)?|proof[ -]of[ -]work)\b', topic, re.I))
                  and not re.search(r'\b(?:data[ -]min(?:ers?|ing)|coal|underground)\b', topic, re.I),
        'cluster-roles': bool(re.search(r'\bphysical servers?\b', source, re.I)
                              and (roles := re.search(r'\bphysical servers?\s*\(([^()]*)\)', source, re.I))
                              and re.search(r'\b\d+\s+managers?\b', roles[1], re.I)
                              and re.search(r'\b\d+\s+workers?\b', roles[1], re.I)
                              and not re.search(r'\b(?:staff|people|personnel|installed|operated|managed by|supervised)\b', source, re.I)),
    }
    selected: list[tuple[str, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, target, domain in _RULES:
        if not enabled[domain]:
            continue
        matches = [m for m in re.finditer(pattern, source, re.I)
                   if not any(start < m.end() and m.start() < end for start, end in occupied)
                   and not (domain == 'degree' and re.search(r'\d[\s-]*$', source[:m.start()]))]
        if matches:
            selected.append((matches[0][0], target))
            occupied.extend((m.start(), m.end()) for m in matches)
    return tuple(selected[:12])
