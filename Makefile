# 宿主 UV_CACHE_DIR 值带引号会让 uv 失败；项目内统一修正
export UV_CACHE_DIR := D:/caches/uv
PYML := uv run --group ml python
PY := uv run python
# Windows 下 PATH 里的 make 可能是 Cygwin 版，无法执行 Docker Desktop 的 sh shim；
# 直接调用 docker.exe 可兼容（其他平台仍用 docker）。
ifeq ($(OS),Windows_NT)
DOCKER ?= docker.exe
else
DOCKER ?= docker
endif

.PHONY: dev langfuse-env langfuse-up langfuse-wait langfuse-down langfuse-logs langfuse-status merge gap-review vectorize eval-merge eval-live test

dev:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

langfuse-env:
	$(PY) -m app.jobs.init_langfuse_env

langfuse-up: langfuse-env
	$(DOCKER) compose -f docker-compose.langfuse.yml up -d
	@echo "Langfuse: http://localhost:3000（首次启动约 1-2 分钟，等待 health 通过）"
	@$(MAKE) --no-print-directory langfuse-wait

langfuse-wait:
	@i=0; until curl -fsS http://localhost:3000/api/public/health >/dev/null 2>&1; do \
		i=$$((i+1)); \
		if [ $$i -ge 60 ]; then echo "Langfuse 未在 120 秒内就绪，请查看 make langfuse-logs"; exit 1; fi; \
		sleep 2; \
	done; echo "Langfuse 已就绪"

langfuse-down:
	$(DOCKER) compose -f docker-compose.langfuse.yml down

langfuse-logs:
	$(DOCKER) compose -f docker-compose.langfuse.yml logs -f --tail=100

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
