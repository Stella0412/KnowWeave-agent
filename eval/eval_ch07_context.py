# eval/eval_ch07_context.py
"""真实服务验收：默认 20 轮不触发；演示配置出现降级/摘要级联。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = Path("log/app.log")


def _events():
    """读取当前日志及轮转备份（.1/.2/.3），解析全部 JSON 事件。"""
    paths = [LOG_PATH]
    for suffix in (".1", ".2", ".3"):
        rotated = Path(str(LOG_PATH) + suffix)
        if rotated.exists():
            paths.append(rotated)
    items = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                items.append(payload)
    return items


def _events_for(session_id: int):
    """按本轮会话 id 过滤事件，避免 10MB 日志轮转把事件拆到备份文件后漏判。"""
    return [e for e in _events() if e.get("conversation_id") == session_id]


_FILLER = (
    "您好，我昨天下午在你们知序下单购买了一个智能定时自动喂食器，商品容量是五升，"
    "适合出差时给猫狗定时出粮，今天上午快递刚刚送到家里，我拆开包装之后发现电源适配器插上插座指示灯完全不亮，"
    "机器按了开机键也没有任何反应，想请你们帮我安排一次售后检测或者直接换新。"
    "另外我上周三还买过一个实木多层猫窝，物流单号显示快递已经揽收，但是连续四天都没有更新任何运输轨迹，"
    "麻烦帮我查一下这单包裹现在到底到哪了，大概什么时候能够派送到家。"
    "喂食器的说明书上写着可以用手机 App 远程设置每天的投喂时间和每次的出粮份量，"
    "我反复尝试了七八次蓝牙配对都失败，App 界面一直提示设备处于离线状态，"
    "让我确认手机和设备是否连接在同一个无线网络下面，可我已经检查过家里的路由器一切正常。"
    "如果最终确认是产品本身存在质量缺陷，我想申请七天无理由退货退款，麻烦说明退款需要满足哪些条件，"
    "需要我上传哪些凭证照片，审核通过之后退款大概需要几个工作日才能原路退回我的账户。"
    "我之前还咨询过猫咪化毛膏和狗狗体外驱虫药的成分与建议用量，当时的客服答复说可以开具电子发票，"
    "我想确认发票抬头和纳税人识别号应该怎么填写，个人抬头和公司抬头之间有什么区别，开票之后多久能收到电子发票。"
    "店里最近有没有适合七岁以上老年柯基犬的关节养护保健品，我家狗狗后腿最近有点使不上劲，"
    "走路速度明显变慢，兽医建议适当补充氨糖和软骨素之类的成分。"
    "另外想问问你们家的猫粮有没有小包装的试用装，我家猫咪肠胃比较敏感，想先买一袋五百克的看看适口性，"
    "顺便确认一下猫粮的生产日期和保质期一般是多少个月。"
    "还有我上次的包裹被放进了小区快递柜，取件码短信没有收到，能不能帮我查一下取件码，或者重新给我发一次。"
    "顺便还想确认一下，如果申请换货的话，寄回去的运费是由你们承担还是我自己出，换货的流程大概要走几天。"
    "你们店铺最近有什么满减活动或者会员日的优惠吗，我想看看能不能赶上。"
    "另外我家还有一只刚满三个月的小猫，想问问幼猫粮和成猫粮能不能混着喂，换粮的时候需要注意什么。"
    "最后想请你们帮我转接人工客服，把喂食器无法开机的具体情况再详细说明一遍，谢谢你们的耐心解答。"
)


def _pad(text: str, min_chars: int) -> str:
    while len(text) < min_chars:
        text += _FILLER
    return text


async def _ask(client, base_url, session_id, message):
    resp = await client.post(
        f"{base_url}/chat/agent",
        json={"session_id": session_id, "message": message},
    )
    resp.raise_for_status()
    return resp.json()


async def run(mode: str, turns: int, base_url: str) -> int:
    async with httpx.AsyncClient(timeout=120.0) as client:
        session_id = None
        # 默认每轮控制在约 1000 token，留出 assistant 开销后仍 <= 1200；
        # 演示每轮更长，确保 20 轮内出现级联。
        pad = 900 if mode == "default" else 1600
        for index in range(turns):
            message = _pad(f"第{index}轮，订单1001的猫爬架进度怎么样？", pad)
            payload = await _ask(client, base_url, session_id, message)
            session_id = str(payload["session_id"])
            print(
                f"turn {index + 1}: session={session_id} "
                f"interrupted={payload.get('interrupted')}"
            )
        events = _events_for(int(session_id))
        if mode == "default":
            demoted = [e for e in events if e.get("event") == "layer1 降级"]
            triggered = [e for e in events if e.get("event") == "summary trigger"]
            assert not demoted, f"默认 20 轮不应降级: {demoted[:1]}"
            assert not triggered, f"默认 20 轮不应触发摘要: {triggered[:1]}"
            print("PASS 默认配置：20 轮无降级、无摘要")
            return 0

        demoted = [e for e in events if e.get("event") == "layer1 降级"]
        triggered = [e for e in events if e.get("event") == "summary trigger"]
        done = [e for e in events if e.get("event") == "summary done"]
        assert demoted, "演示配置未出现 layer1 降级"
        assert triggered, "演示配置未出现 summary trigger"
        assert done, "演示配置未出现 summary done"
        answer = await _ask(
            client, base_url, session_id, "最开始那个订单后来怎么说？"
        )
        text = answer.get("answer") or ""
        assert "1001" in text or "猫爬架" in text, f"摘要未回答问题: {text}"
        print("PASS 演示配置：降级/触发/完成级联，摘要可回答早期订单")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "demo"], default="default")
    parser.add_argument("--turns", type=int, default=22)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    return asyncio.run(run(args.mode, args.turns, args.base_url))


if __name__ == "__main__":
    raise SystemExit(main())
