from app.jobs.gap_review import parse_args


def test_parse_args_actions_are_mutually_exclusive():
    args = parse_args(["--list", "--status", "待处理"])
    assert args.list is True
    assert args.status == "待处理"
    args = parse_args(["--answer", "3", "--question-file", "q.txt",
                       "--answer-file", "a.txt"])
    assert args.answer == 3
    assert args.question_file == "q.txt"
    assert args.answer_file == "a.txt"


def test_parse_args_metrics_and_show():
    assert parse_args(["--metrics"]).metrics is True
    assert parse_args(["--show", "7"]).show == 7
