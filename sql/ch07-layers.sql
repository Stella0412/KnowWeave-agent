ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT NULL;

-- layer1_from_msg_id 是层 1 第一条用户消息 id，层 1 包含该条：
-- id >= layer1_from_msg_id 属层1
-- summary_upto_msg_id < id < layer1_from_msg_id 属层2（按整轮判定）
-- id <= summary_upto_msg_id 已进摘要（按整轮判定）
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id BIGINT NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT NOT NULL,
  seq INT NOT NULL,
  from_msg_id BIGINT NOT NULL,
  upto_msg_id BIGINT NOT NULL,
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conversation_seq (conversation_id, seq),
  KEY idx_conversation_upto (conversation_id, upto_msg_id),
  CONSTRAINT fk_summary_conversation
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
