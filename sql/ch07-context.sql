ALTER TABLE conversations
  ADD COLUMN summary TEXT NULL,
  ADD COLUMN summary_upto_msg_id BIGINT NULL;
