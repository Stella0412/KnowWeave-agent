-- =============================================================
-- ch09 · 可观测性与成本 · 每轮链路指标
-- 本章新建:turn_metrics(每轮 route/intent/状态/token/成本/耗时)
-- 扩列:tool_audit_logs(turn_token 按轮关联工具调用)
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT 按 utf8mb4 解析
SET NAMES utf8mb4;

CREATE TABLE turn_metrics (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT NOT NULL,
  turn_token VARCHAR(64) NOT NULL,
  trace_id VARCHAR(64) NULL,
  route VARCHAR(32) NULL,
  intent VARCHAR(32) NULL,
  status ENUM('ok','error','interrupted','cancelled') NOT NULL,
  error_type VARCHAR(64) NULL,
  llm_calls INT NOT NULL DEFAULT 0,
  prompt_tokens INT NOT NULL DEFAULT 0,
  completion_tokens INT NOT NULL DEFAULT 0,
  total_tokens INT NOT NULL DEFAULT 0,
  usage_estimated TINYINT(1) NOT NULL DEFAULT 0,
  model_usage JSON NULL,
  stages JSON NULL,
  cost_cny DECIMAL(12,6) NOT NULL DEFAULT 0,
  latency_ms INT NOT NULL DEFAULT 0,
  first_token_ms INT NULL,
  refusal_source VARCHAR(32) NULL,
  citations_count INT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_turn_token (turn_token),
  KEY idx_turn_created (created_at),
  KEY idx_turn_conversation (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE tool_audit_logs
  ADD COLUMN turn_token VARCHAR(64) NULL AFTER conversation_id,
  ADD KEY idx_turn_token (turn_token);
