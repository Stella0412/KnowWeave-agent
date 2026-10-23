"""独立物流查询 mock MCP Server（Streamable HTTP，官方 MCP Python SDK）。

纯 mock 数据：随机生成运单轨迹，不接真实系统、不建业务表、不导入 app 任何模块。
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
_ANCHOR = datetime(2026, 10, 20, 8, 0, 0, tzinfo=timezone.utc)


def make_tracking(tracking_no: str, rng: random.Random | None = None) -> dict:
    """生成一条 mock 物流轨迹（纯函数，注入固定 rng 即可重复）。"""
    rng = rng if rng is not None else random.Random()
    step = rng.randint(0, len(_TRACKING_FLOW) - 1)
    event_count = step + rng.randint(1, 3)

    events = []
    when = _ANCHOR + timedelta(hours=rng.randint(0, 12))
    for i in range(event_count):
        events.append({
            "time": when.isoformat(),
            "status_code": _TRACKING_FLOW[min(i, step)],
        })
        when = when + timedelta(hours=rng.randint(1, 24), minutes=rng.randint(0, 59))

    return {
        "tracking_no": tracking_no,
        "carrier": rng.choice(_CARRIERS),
        "status_code": _TRACKING_FLOW[step],
        "events": events,
        "mock": True,
    }


def query_logistics(tracking_no: str) -> dict:
    """按运单号查询物流轨迹（mock 数据）。"""
    return make_tracking(tracking_no)


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
