# ch10 · 低置信度问题多标签主题分类器

本章把数据飞轮里累积的低置信度问题批量归入 17 个权威类目，回答“下一批知识该补哪一块”。
模型是 `hfl/chinese-roberta-wwm-ext` 全参微调的多标签分类器，旁路运行：实时聊天主链路不调用
分类服务，只有批量任务在池子里攒够问题后调用一次推理服务，把结果写进 `topic_classifications`。
以下命令默认在仓库根目录（D:/knowweave）执行，Python 3.14，依赖用 `uv run` 管理。

## 1. 架构与旁路边界

```text
实时聊天 /chat
  └─ ChatService → 检索/生成 → 低置信度问题入池
                             （不调用主题分类）

旁路批处理（人工或定时触发）
  app.jobs.topic_classify
    ├─ 选取未归类根问题（默认不足 batch_size 则跳过）
    ├─ HTTP POST /classify ──► ml.service.app（ONNX Runtime，127.0.0.1:8200）
    ├─ 校验返回条数，单事务 upsert
    └─ topic_classifications（question_id 唯一，labels 为 JSON 多标签）

后台页面（只读聚合）
  /topics               主题分布、原始/加权、多标签组合
  /topics/questions     单类目问题列表
  /acceptance           训练/评测/导出门禁
  /admin                管理入口
```

边界：分类服务与聊天服务是两个进程；聊天主路径没有新增同步调用，也不依赖 ONNX 模型在线。
批处理失败时整批不写，下一次可重跑；`question_id` 唯一键保证幂等。

## 2. 17 类类目与近邻边界

权威顺序（`config/ch10_taxonomy.json`，`其他` 固定最后）：

| # | 类目 | # | 类目 | # | 类目 |
|---|---|---|---|---|---|
| 1 | 退换货 | 7 | 优惠活动 | 13 | 保修维修 |
| 2 | 物流 | 8 | 价保 | 14 | 账号 |
| 3 | 尺码 | 9 | 支付 | 15 | 会员积分 |
| 4 | 发票 | 10 | 订单修改 | 16 | 评价 |
| 5 | 质量问题 | 11 | 库存补货 | 17 | 其他 |
| 6 | 运费 | 12 | 商品信息 | | |

三条近邻边界：

- **修归保修维修，退归退换货**：问“能不能修/换新”是保修维修；问“能不能退/退款”是退换货。
- **运费管钱，物流管货**：费用谁承担、能否报销是运费；到哪了、几天到、丢件是物流。
- **价保是补差价，优惠活动是券和满减**：买后降价补差是价保；领券、满减、叠加用是优惠活动。

`其他` 不是排他兜底：同一句既有“转人工/投诉”等泛诉求，又有具体主题时，允许与其他标签共存
（训练数据中有 25 条这样的样本）。

## 3. 数据口径

数据来自用户提供的 `ch10-dataset/`，只读使用，不做合成、不改写：

| 切分 | 行数 | 多标签行 |
|---|---:|---:|
| train | 2594 | 436 |
| val | 161 | 30 |
| test | 161 | 20 |
| 合计 | 2916 | 486 |

- 审计命令：`make topic-audit`；manifest 在 `ch10-dataset/manifest.json`。
- 17 类全覆盖；测试集每类 support 9–14，单类指标方差较大。
- PII 规则扫描 0 命中；跨切分完全重复为硬错误，训练集内部 2 组完全重复仅告警。
- 本章未做 200 条人工抽审：数据视为已确认；人工复核集中在测试错例与真实池归类结果。

## 4. 训练

固定技术选择：HFL `hfl/chinese-roberta-wwm-ext`，全参微调，不用 LoRA/QLoRA，不训练 embedding。

| 项 | 值 |
|---|---|
| 损失 | `BCEWithLogitsLoss` + 每类 `pos_weight`（上限 10） |
| 最大长度 | 128 |
| batch / 梯度累积 | 16 / 2 |
| 学习率 | 2e-5 |
| warmup | 总步数 10% |
| epochs | 4 |
| 设备 | CPU（本机 torch 为 `2.14.1+cpu`，未启用 GPU） |
| 选择指标 | 验证集 macro-F1，`load_best_model_at_end` |

验证集 macro-F1：epoch 1 `0.7052` → epoch 2 `0.9096` → epoch 3 `0.9313` → epoch 4 `0.9437`。
训练产物在 `data/ch10/model/run-20261026-234705/`：`best/`、`metrics.json`、`label_map.json`、
`train_config.json`、`manifest.json`、`thresholds.json`；`latest.txt` 指向该 run。

## 5. 阈值与测试集指标

每类阈值在验证集上按 F1 网格搜索（0.20–0.70，步长 0.05），写入 `thresholds.json`；推理时保留所有
过阈值的标签，全不过阈值时取 argmax，保证至少一个标签。最终阈值示例：发票 0.30、运费 0.40、
会员积分 0.40、物流 0.55、其他 0.60、退换货 0.70。

测试集 161 条结果：

| 指标 | 值 |
|---|---:|
| macro-F1 | 0.9558 |
| micro-F1 | 0.9558 |
| 完全匹配率 | 0.9193 |
| 平均预测标签数 | 1.12 |

较弱类目（完整每类 P/R/F1 与 2×2 混淆矩阵见 `eval/reports/ch10-topic-2026-10-26.md` 与
`/acceptance/eval`）：

| 类目 | support | P | R | F1 |
|---|---:|---:|---:|---:|
| 商品信息 | 10 | 0.8889 | 0.8000 | 0.8421 |
| 退换货 | 14 | 0.9231 | 0.8571 | 0.8889 |
| 其他 | 10 | 1.0000 | 0.8000 | 0.8889 |
| 质量问题 | 9 | 0.8182 | 1.0000 | 0.9000 |
| 尺码 | 10 | 0.8333 | 1.0000 | 0.9091 |

黄金用例 20 条通过 17 条。未通过的三条：

| 用例 | 期望 | 预测 | 判断 |
|---|---|---|---|
| 收到就破了，想退货 | 质量问题 / 退换货 | 退换货 | 漏“质量问题” |
| 订单下错了，还没发货能取消吗 | 订单修改 | 物流 / 订单修改 | 多“物流” |
| 这个和图片颜色不一样 | 质量问题 | 退换货 | 边界混淆 |

多诉求验收样例：`买大了想退` → `退换货 0.9095` + `尺码 0.9258`；
`运费太贵了，而且物流三天没动` → `物流 0.8744` + `运费 0.8571`。

## 6. 错例人工复核

评测按每类最多 2 条 FP/FN 导出 16 行（13 个唯一样本）到 `data/ch10/eval/errors_to_audit.csv`。
逐条复核结论：

| 样本 | 真实 | 预测 | 类型 | 复核结论 |
|---|---|---|---|---|
| test-7 刚下单…改订单价格还是申请价保 | 价保 / 订单修改 | 价保 | FN 订单修改 | 模型漏标，句子明确同时涉及改价与价保 |
| test-20 客服几点有人…优惠券能否叠加 | 其他 / 优惠活动 | 优惠活动 | FN 其他 | 泛诉求与具体主题共存，模型漏“其他” |
| test-27 会员专享价…等级 | 会员积分 | 优惠活动 / 会员积分 | FP 优惠活动 | “专享价”触发优惠活动，边界误判 |
| test-38 猫抓板用旧了可以回收吗 | 其他 | 商品信息 | FP/FN | 标注边界争议；“回收”可归其他，维持原标签 |
| test-60 猫粮颗粒大不大…牙口 | 商品信息 | 尺码 | FP/FN | 尺码泛化到“颗粒大小”，边界误判 |
| test-62 猫粮颗粒…成年猫嚼得动吗 | 商品信息 | 尺码 | FP/FN | 同上 |
| test-98 猫砂盆揽收后没动…丢件 | 物流 | 物流 / 质量问题 | FP 质量问题 | “丢件”被误读为质量，实为物流 |
| test-99 配件坏了能寄新的…补发物流几天 | 保修维修 / 物流 | 物流 | FN 保修维修 | 漏“坏了/寄新的” |
| test-140 退货快递费能报销吗 | 运费 | 物流 / 运费 | FP 物流 | “快递”触发物流，费用问题应只归运费 |
| test-141 退货运费谁承担 | 运费 | 退换货 / 运费 | FP 退换货 | “退货”触发退换货，问的是费用 |
| test-152 退货后积分会退回吗 | 会员积分 / 退换货 | 会员积分 | FN 退换货 | 跨句“退货”漏标 |
| test-154 退货了抵扣积分会退回吗 | 会员积分 / 退换货 | 会员积分 | FN 退换货 | 同上 |
| test-156 味道太大…申请退款 | 退换货 | 退换货 / 质量问题 | FP 质量问题 | “味道”触发质量；原标签只保留退换货 |

**标签修正记录**：未修改训练/验证/测试集任何标签。上表 test-38、test-60/62、test-156 属于标注边界
争议，保留原标签并记录理由；其余 9 条为模型漏标/多标。真实池归类中有两条需人工留意：
`表单没有显示` → 发票（上下文不足）、`我要退另一张订单…` → 订单修改（疑似漏退换货），
未手工改库，留待飞轮审核页处理。

## 7. ONNX 导出与推理服务

```bash
make topic-export    # 导出 run/onnx，并原子替换 data/ch10/model/current
make topic-serve     # 127.0.0.1:8200
make topic-smoke ARGS="--text 买大了想退"
```

- 导出：opset 17、动态 batch/sequence；PyTorch 与 ONNX Runtime 最大绝对偏差
  `1.907e-06`，低于 `1e-3` 门禁。
- 服务：FastAPI + ONNX Runtime CPU，`GET /health`、`GET /model`、`POST /classify`；
  `texts` 限制 1–128 条且必须全为字符串。
- 线上模型版本来自 run manifest：`run-20261026-234705`。

## 8. 批量归类与后台页面

```bash
# 池子不足 batch_size 时默认跳过；验收时显式全量跑一次
make topic-classify ARGS="--reclassify-all --exclude-pattern 第九章闭环测试"
```

真实池现有 12 条根问题，其中 1 条是第九章闭环测试留下的痕迹，按模式排除；本次归类 11 条，
写入 MySQL `topic_classifications`。分布：退换货 3、物流 2、订单修改 2、发票/优惠活动/支付/
商品信息/保修维修各 1；加权总量 25（按 `duplicate_count`）。

页面：

| 页面 | URL |
|---|---|
| 后台管理首页 | `/admin` |
| 验收总览 | `/acceptance` |
| 评测详情 | `/acceptance/eval` |
| 数据产物 | `/acceptance/data` |
| 错例复核 | `/acceptance/errors` |
| 主题分布 | `/topics` |
| 类目问题列表 | `/topics/questions?label=商品信息` |

页面全部使用原生 HTML/CSS/JS，不引外部图表库，DOM 文本一律 `textContent`；主题分布支持原始/加权
切换、点击类目加载样例、多标签组合 Top 10；验收页把测试指标、ONNX 对齐、黄金用例、数据集产物
四项门禁直接列出。

## 9. 验收结论

| 验收项 | 结果 |
|---|---|
| 测试集每类 F1 与混淆矩阵报告可跑 | `make topic-eval` 生成报告，macro-F1 0.9558，17 类 2×2 矩阵齐全 |
| 真实池批量归类 + 后台分布页计数 | 11 条真实根问题入库；`/topics` 显示 17 类计数与组合 |
| 单句多意图 | `买大了想退` 同时命中 `退换货` 与 `尺码` |

回归：默认套件 `851 passed, 8 skipped`；ML 套件见下节命令。`python -m compileall -q app ml tests eval`
无错误。

## 10. 复现命令

```bash
make topic-audit     # 只读审计 ch10-dataset，写 manifest
make topic-train     # 全参微调，CPU 约 70 分钟
make topic-tune      # 验证集阈值 → run/thresholds.json
make topic-eval      # 测试集指标、报告、错例 CSV
make topic-export    # ONNX + parity 门禁 + promote current
make topic-serve     # 终端 A：推理服务 8200
make topic-classify ARGS="--reclassify-all --exclude-pattern 第九章闭环测试"
make topic-smoke ARGS="--text 买大了想退"
make dev             # 终端 B：后台 8000
```

MySQL 建表：`docker.exe exec -i knowweave-mysql mysql -uroot -p knowweave < sql/ch10-topic.sql`
（按提示输入本地 root 密码，不把密码写进仓库）。
现有 `low_confidence_questions.id` 是 signed `BIGINT`，DDL 的 `id/question_id` 已对齐为 signed，
否则外键报 ERROR 3780。

ML 测试：

```bash
UV_CACHE_DIR=D:/caches/uv uv run --group ml pytest tests/test_ch10_dataset.py tests/test_ch10_decision.py \
  tests/test_ch10_train.py tests/test_ch10_eval.py tests/test_ch10_onnx.py \
  tests/test_ch10_service.py tests/test_ch10_topic_job.py tests/test_ch10_topics_api.py \
  tests/test_ch10_acceptance.py -q
```

## 11. 局限

- 测试集每类只有 9–14 条，单类 F1 与混淆矩阵方差大，不能外推为线上全量分布。
- 数据以合成/构造句为主，真实池只有 11 条有效根问题；真实域指标尚不足以下结论。
- 长尾与分布漂移：新类目、活动话术、跨域组合可能不在训练分布内；阈值基于验证集，存在过拟合风险。
- 多标签边界仍集中在“退货/运费/物流”“商品信息/尺码”“质量问题/退换货”，需要持续用飞轮错例回流。
- 旁路批处理是“攒够再跑”，不是实时分类；池子不足时默认不写，避免低频批次浪费推理。
