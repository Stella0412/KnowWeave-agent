import logging


def test_turn_logger_is_visible_at_info():
    import app.main  # noqa: F401  导入即完成日志配置

    logger = logging.getLogger("knowweave.turn")
    assert logger.getEffectiveLevel() == logging.INFO
    assert logger.hasHandlers()
