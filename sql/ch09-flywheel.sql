-- =============================================================
-- ch09 · 数据飞轮 · 反馈落库 + 低置信度池扩列
-- 本章新建:answer_feedback(👍/👎 反馈落库,message_id 唯一)
-- 扩列:low_confidence_questions(指纹/归并/状态/快照/处置)
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT 按 utf8mb4 解析
SET NAMES utf8mb4;

CREATE TABLE answer_feedback (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT NOT NULL,
  message_id BIGINT NOT NULL,
  rating ENUM('up','down') NOT NULL,
  comment VARCHAR(300) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_feedback_message (message_id),
  KEY idx_feedback_conversation (conversation_id),
  KEY idx_feedback_rating_created (rating, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE low_confidence_questions
  ADD COLUMN message_id BIGINT NULL AFTER conversation_id,
  ADD COLUMN fingerprint CHAR(64) NOT NULL DEFAULT '' AFTER raw_question,
  ADD COLUMN merged_into_id BIGINT NULL AFTER fingerprint,
  ADD COLUMN duplicate_count INT NOT NULL DEFAULT 1 AFTER merged_into_id,
  ADD COLUMN status ENUM('待处理','已归并','已关闭','已忽略')
      NOT NULL DEFAULT '待处理' AFTER duplicate_count,
  ADD COLUMN citations JSON NULL AFTER reason,
  ADD COLUMN knowledge_chunk_id BIGINT NULL AFTER citations,
  ADD COLUMN resolution VARCHAR(500) NULL AFTER knowledge_chunk_id,
  ADD COLUMN last_seen_at DATETIME NULL AFTER created_at,
  ADD UNIQUE KEY uk_message_source (message_id, source),
  ADD KEY idx_fingerprint_root (fingerprint, merged_into_id),
  ADD KEY idx_status_last_seen (status, last_seen_at),
  ADD KEY idx_merged_into (merged_into_id);

-- 先把历史行对齐到 created_at，再收紧 NOT NULL
UPDATE low_confidence_questions SET last_seen_at = created_at WHERE last_seen_at IS NULL;
ALTER TABLE low_confidence_questions
  MODIFY COLUMN last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP;
