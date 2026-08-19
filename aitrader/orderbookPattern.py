def analyze_orderbook_pattern(orderbook_data):
    """
    업비트 호가창(orderbook) 데이터를 분석하여 진입 여부와 패턴을 반환합니다.

    :param orderbook_data: 업비트 웹소켓/API에서 받은 orderbook 딕셔너리
    :return: (상태코드, 설명)
             상태코드: 'DO_NOT_ENTER'(절대 진입 금지), 'MONITOR'(돌파 감시), 'WAIT'(일반 상태)
    """
    total_ask_size = orderbook_data['total_ask_size']
    total_bid_size = orderbook_data['total_bid_size']
    units = orderbook_data['orderbook_units']  # 1호가 ~ 15호가 리스트

    # 0. 데이터 오류 방지
    if total_ask_size == 0 or total_bid_size == 0:
        return 'DO_NOT_ENTER', '호가창 데이터 오류 또는 거래 정지 상태'

    # 1. 극단적 불균형 필터 (어느 한쪽이 5배 이상 많으면 조작/위험 상태)
    ratio = max(total_ask_size, total_bid_size) / min(total_ask_size, total_bid_size)
    if ratio >= 5.0:
        return 'DO_NOT_ENTER', f'극단적 호가 불균형 (비율 {ratio:.1f}배) - 조작 위험'

    # 2. 호가대역 분리 (근접 호가 vs 먼 호가)
    # 1~5호가는 현재가 근처의 '실제 방어/저항 물량'
    # 6~15호가는 현재가와 먼 '대기 물량 (또는 허매수/허매도)'
    near_ask_size = sum(u['ask_size'] for u in units[:5])
    far_ask_size = sum(u['ask_size'] for u in units[5:])

    near_bid_size = sum(u['bid_size'] for u in units[:5])
    far_bid_size = sum(u['bid_size'] for u in units[5:])

    # ---------------------------------------------------------
    # [패턴 1] 하락 전조 / 허매수 (Fake Bid) 패턴 - (2번째, 4번째 이미지)
    # 전체 매수 잔량의 75% 이상이 저 밑(6~15호가)에 깔려있고,
    # 정작 현재가 근처(1~5호가)의 매수벽은 매도벽보다 얇을 때
    # ---------------------------------------------------------
    if (far_bid_size / total_bid_size) > 0.75 and near_bid_size < near_ask_size:
        return 'DO_NOT_ENTER', '허매수(Fake Bid) 패턴 감지 - 하락 위험 매우 높음'

    # ---------------------------------------------------------
    # [패턴 2] 설거지 / 개미 꼬시기 패턴
    # 매수 총잔량이 매도 총잔량보다 2배 이상 많은데,
    # 현재가 근처에만 매수벽이 비정상적으로 두꺼운 경우 (안심시키고 던지기)
    # ---------------------------------------------------------
    if (total_bid_size / total_ask_size) > 2.0 and near_bid_size > (near_ask_size * 2):
        return 'DO_NOT_ENTER', '설거지(매수 유도) 패턴 의심 - 고점 물림 위험'

    # ---------------------------------------------------------
    # [패턴 3] 상승 전조 / 억누르기 (매집) 패턴 - (1번째, 3번째 이미지)
    # 매도 총잔량이 매수 총잔량보다 1.5배 ~ 4배 사이로 많고,
    # 현재가 근처(1~5호가)의 매도벽이 두껍게 짓누르고 있을 때
    # ---------------------------------------------------------
    if 1.5 <= (total_ask_size / total_bid_size) <= 4.0 and near_ask_size > near_bid_size:
        return 'MONITOR', '매도벽 억누르기(매집) 패턴 - 돌파 시 매수 대기(관심 종목)'

    # 위 패턴에 해당하지 않는 일반적인 호가창
    return 'WAIT', '특이사항 없음 - 보조지표(RSI, 거래량 등)에 따라 판단'


# ==========================================
# 봇 적용 예시 (모니터링 로직)
# ==========================================
def on_orderbook_received(orderbook_data):
    status, reason = analyze_orderbook_pattern(orderbook_data)

    if status == 'DO_NOT_ENTER':
        # 이 코인은 RSI가 아무리 좋아도 절대 사지 않고 패스함
        # print(f"[패스] {reason}")
        pass

    elif status == 'MONITOR':
        # 호가창이 상승 전조를 보임.
        # 이때 체결 데이터(trade)를 감시하다가 누군가 시장가 대량 매수를 터뜨리면 즉시 진입!
        print(f"[감시 시작] {reason}")
        # check_breakout_and_buy(orderbook_data['market'])

    elif status == 'WAIT':
        # 호가창은 평범함. 기존에 설정한 RSI, 5분봉 변동폭 등의 조건이 맞으면 진입
        # check_technical_indicators()
        pass
