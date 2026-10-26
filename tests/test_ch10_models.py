from app.db.models import TopicClassification


def test_topic_classification_table_and_columns():
    assert TopicClassification.__tablename__ == "topic_classifications"
    columns = TopicClassification.__table__.columns
    assert set(columns.keys()) == {"id", "question_id", "labels", "classified_at"}
    assert columns["labels"].nullable is False
    assert columns["question_id"].nullable is False
