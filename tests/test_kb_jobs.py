from app.jobs.build_kb import parse_args as parse_build_args
from app.jobs.mine_qa import parse_args as parse_mine_args
from app.jobs.vectorize_kb import parse_args as parse_vec_args


def test_build_kb_args_defaults():
    args = parse_build_args([])
    assert args.rebuild is False


def test_vectorize_kb_args_defaults():
    args = parse_vec_args([])
    assert args.batch_size == 200


def test_mine_qa_args_defaults():
    args = parse_mine_args([])
    assert args.batch_size is None
    assert args.run_id is None
