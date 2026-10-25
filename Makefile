PY := uv run python

.PHONY: dev langfuse-env langfuse-up langfuse-wait langfuse-down langfuse-logs langfuse-status merge gap-review vectorize eval-merge eval-live test

dev:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

langfuse-env:
	$(PY) -m app.jobs.init_langfuse_env

langfuse-up: langfuse-env
	docker compose -f docker-compose.langfuse.yml up -d
	@echo "Langfuse: http://localhost:3000（首次启动约 1-2 分钟，等待 health 通过）"
	@$(MAKE) --no-print-directory langfuse-wait

langfuse-wait:
	@i=0; until curl -fsS http://localhost:3000/api/public/health >/dev/null 2>&1; do \
		i=$$((i+1)); \
		if [ $$i -ge 60 ]; then echo "Langfuse 未在 120 秒内就绪，请查看 make langfuse-logs"; exit 1; fi; \
		sleep 2; \
	done; echo "Langfuse 已就绪"

langfuse-down:
	docker compose -f docker-compose.langfuse.yml down

langfuse-logs:
	docker compose -f docker-compose.langfuse.yml logs -f --tail=100

langfuse-status:
	@curl -fsS http://localhost:3000/api/public/health && echo

merge:
	$(PY) -m app.jobs.low_conf_merge $(ARGS)

gap-review:
	$(PY) -m app.jobs.gap_review $(if $(ARGS),$(ARGS),--list)

vectorize:
	$(PY) -m app.jobs.vectorize_kb

eval-merge:
	$(PY) eval/eval_ch09_merge.py

eval-live:
	CH09_LIVE=1 $(PY) -m pytest -q tests/test_ch09_live.py

test:
	$(PY) -m pytest -q
