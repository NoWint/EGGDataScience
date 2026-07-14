"""测试 experiments 表新列"""
import sqlite3
import tempfile
from pathlib import Path
from app import database


def test_experiments_has_new_columns(tmp_path, monkeypatch):
    """experiments 表应包含 eeg_path, source, analysis_status 列"""
    # 用临时数据库
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(database, "DB_DIR", tmp_path)
    database.init_db()

    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(experiments)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "eeg_path" in columns
    assert "source" in columns
    assert "analysis_status" in columns
