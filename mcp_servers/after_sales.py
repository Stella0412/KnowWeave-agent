"""独立售后查询 mock MCP Server（Streamable HTTP，官方 MCP Python SDK）。

纯 mock 数据：随机生成保修结论与退货进度，不接真实系统、不建业务表、不导入 app。
status_code 使用服务端内部 ASCII 码（in_warranty / out_of_warranty /
pending / in_transit / refunded），中文业务标签由 Agent 侧可信格式化器翻译。
"""

import argparse
import random
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8102

_RETURN_FLOW = ("pending", "in_transit", "refunded")
_WARRANTY_ANCHOR = datetime(2026, 10, 23)
_RETURN_ANCHOR = datetime(2026, 10, 20, 9, 0, 0, tzinfo=timezone.utc)


def make_warranty(product_name: str, rng: random.Random | None = None) -> dict:
    """生成一条 mock 保修结论（纯函数，注入固定 rng 即可重复）。"""
    rng = rng if rng is not None else random.Random()
    in_warranty = rng.random() < 0.7
    if in_warranty:
        warranty_until = (_WARRANTY_ANCHOR + timedelta(days=rng.randint(30, 730))).date().isoformat()
    else:
        warranty_until = (_WARRANTY_ANCHOR - timedelta(days=rng.randint(1, 365))).date().isoformat()

    return {
        "product_name": product_name,
        "in_warranty": in_warranty,
        "warranty_until": warranty_until,
        "status_code": "in_warranty" if in_warranty else "out_of_warranty",
        "mock": True,
    }


def make_return_progress(return_no: str, rng: random.Random | None = None) -> dict:
    """生成一条 mock 退货进度（纯函数，注入固定 rng 即可重复）。"""
    rng = rng if rng is not None else random.Random()
    step = rng.randint(0, len(_RETURN_FLOW) - 1)
    step_count = step + rng.randint(1, 3)

    progress = []
    when = _RETURN_ANCHOR + timedelta(hours=rng.randint(0, 12))
    for i in range(step_count):
        progress.append({
            "time": when.isoformat(),
            "status_code": _RETURN_FLOW[min(i, step)],
        })
        when = when + timedelta(hours=rng.randint(1, 24), minutes=rng.randint(0, 59))

    return {
        "return_no": return_no,
        "status_code": _RETURN_FLOW[step],
        "progress": progress,
        "mock": True,
    }


def query_warranty(product_name: str) -> dict:
    """按商品名称查询保修状态（mock 数据）。"""
    return make_warranty(product_name)


def query_return_progress(return_no: str) -> dict:
    """按退货单号查询退货进度（mock 数据）。"""
    return make_return_progress(return_no)


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> FastMCP:
    """构建 after_sales FastMCP Server；host/port 可覆盖，不在此绑定 socket。"""
    mcp = FastMCP("after_sales", host=host, port=port,
                  stateless_http=True, json_response=True)
    mcp.tool()(query_warranty)
    mcp.tool()(query_return_progress)
    return mcp


mcp = create_server()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="after_sales mock MCP server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    create_server(host=args.host, port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
