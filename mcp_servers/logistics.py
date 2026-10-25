"""独立物流查询 mock MCP Server（Streamable HTTP，官方 MCP Python SDK）。

纯 mock 数据：按运单号确定性生成运单轨迹，不接真实系统、不建业务表、不导入 app 任何模块。
status_code 使用服务端内部 ASCII 码（pending_pickup / in_transit / delivered），
中文业务标签由 Agent 侧可信格式化器翻译，本 Server 不产出中文状态契约。
"""

import argparse
import random
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8101

_CARRIERS = ("顺丰速运", "中通快递", "圆通速递", "韵达快递")
_TRACKING_FLOW = ("pending_pickup", "in_transit", "delivered")
# 演示订单的签收距今天数，与 app/db/seed.py 的 signed_days_ago 保持一致。
_DEMO_SIGNED_DAYS_AGO = {"SF1001CN": 3, "SF1002CN": 10, "SF1005CN": 2}
_DEMO_IN_TRANSIT = {"SF1003CN"}


def _parse_signed_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def make_tracking(
    tracking_no: str,
    rng: random.Random | None = None,
    *,
    now: datetime | None = None,
    signed_at: str | None = None,
) -> dict:
    """生成一条 mock 物流轨迹（纯函数，注入固定 rng/now 即可重复）。

    不传 rng 时按运单号播种 —— 生产路径同一运单号每次返回相同轨迹。
    状态严格按 pending_pickup → in_transit → delivered 推进，签收后不再追加。
    传入 signed_at（订单签收时间）时签收节点对齐该时间；否则演示运单按
    seed 的 signed_days_ago 相对当前日期生成。
    """
    rng = rng if rng is not None else random.Random(tracking_no)
    # 截断到当天零点：同一天内重复查询结果稳定，日期仍相对「今天」。
    now = (now or datetime.now(timezone.utc)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    signed = _parse_signed_at(signed_at)

    if signed is not None or tracking_no in _DEMO_SIGNED_DAYS_AGO:
        step = len(_TRACKING_FLOW) - 1
    elif tracking_no in _DEMO_IN_TRANSIT:
        step = 1
    else:
        step = rng.randint(0, len(_TRACKING_FLOW) - 1)

    if signed is not None:
        last_time = signed
    elif step == 2:
        days_ago = _DEMO_SIGNED_DAYS_AGO.get(tracking_no, rng.randint(1, 5))
        last_time = now - timedelta(days=days_ago)
    elif step == 1:
        last_time = now - timedelta(
            hours=rng.randint(1, 12), minutes=rng.randint(0, 59)
        )
    else:
        last_time = now - timedelta(
            hours=rng.randint(1, 6), minutes=rng.randint(0, 59)
        )

    times = [last_time]
    for _ in range(step):
        gap = timedelta(hours=rng.randint(3, 20), minutes=rng.randint(0, 59))
        times.append(times[-1] - gap)
    times.reverse()

    events = [
        {"time": when.isoformat(), "status_code": _TRACKING_FLOW[index]}
        for index, when in enumerate(times)
    ]
    return {
        "tracking_no": tracking_no,
        "carrier": rng.choice(_CARRIERS),
        "status_code": _TRACKING_FLOW[step],
        "events": events,
        "mock": True,
    }


def query_logistics(tracking_no: str, signed_at: str | None = None) -> dict:
    """按运单号查询物流轨迹（mock 数据）。signed_at 可选，用于与订单签收时间对齐。"""
    return make_tracking(tracking_no, signed_at=signed_at)


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> FastMCP:
    """构建 logistics FastMCP Server；host/port 可覆盖，不在此绑定 socket。"""
    mcp = FastMCP("logistics", host=host, port=port,
                  stateless_http=True, json_response=True)
    mcp.tool()(query_logistics)
    return mcp


mcp = create_server()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="logistics mock MCP server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    create_server(host=args.host, port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
