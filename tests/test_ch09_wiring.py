from app.main import app


def _route_paths(routes) -> set[str]:
    """收集全部已注册路径；兼容 FastAPI 0.143 把 include_router 存成 _IncludedRouter。"""
    paths: set[str] = set()
    for route in routes:
        if hasattr(route, "path"):
            paths.add(route.path)
        original = getattr(route, "original_router", None)
        if original is not None:
            paths |= _route_paths(getattr(original, "routes", None) or [])
    return paths


def test_ch09_routes_are_registered():
    paths = _route_paths(app.routes)
    assert {"/api/feedback", "/review", "/review/api/queue",
            "/observability", "/observability/api/summary"} <= paths
