"""
EEGDataScience 数据库层
SQLite 持久化，管理被试与实验记录
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# 数据库文件路径
DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "eeg.db"


def get_connection() -> sqlite3.Connection:
    """获取数据库连接（自动创建目录和表）"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # 行以字典方式访问
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_connection()
    cursor = conn.cursor()

    # 被试表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # 实验记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            condition TEXT NOT NULL,
            date TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        )
    """)

    # 迁移：为 experiments 表补充新列（向后兼容，已有列则跳过）
    cursor.execute("PRAGMA table_info(experiments)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    migrations = [
        ("eeg_path", "TEXT"),
        ("source", "TEXT DEFAULT 'upload'"),
        ("analysis_status", "TEXT DEFAULT 'pending'"),
    ]
    for col_name, col_def in migrations:
        if col_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE experiments ADD COLUMN {col_name} {col_def}"
            )

    conn.commit()
    conn.close()


# ========== 被试 CRUD ==========
def create_subject(code: str, age: Optional[int] = None, gender: Optional[str] = None) -> Dict[str, Any]:
    """创建被试"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO subjects (code, age, gender) VALUES (?, ?, ?)",
            (code, age, gender)
        )
        conn.commit()
        subject_id = cursor.lastrowid
        subject = get_subject(subject_id)
        conn.close()
        return subject
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"被试编号已存在: {code}")


def get_subject(subject_id: int) -> Optional[Dict[str, Any]]:
    """获取单个被试"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_subjects() -> List[Dict[str, Any]]:
    """列出所有被试"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_subject(subject_id: int, code: Optional[str] = None,
                   age: Optional[int] = None, gender: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """更新被试信息"""
    conn = get_connection()
    cursor = conn.cursor()

    fields = []
    values = []
    if code is not None:
        fields.append("code = ?")
        values.append(code)
    if age is not None:
        fields.append("age = ?")
        values.append(age)
    if gender is not None:
        fields.append("gender = ?")
        values.append(gender)

    if not fields:
        conn.close()
        return get_subject(subject_id)

    values.append(subject_id)
    try:
        cursor.execute(f"UPDATE subjects SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
        result = get_subject(subject_id)
        conn.close()
        return result
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"被试编号已存在: {code}")


def delete_subject(subject_id: int) -> bool:
    """删除被试（级联删除实验记录）"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ========== 实验记录 CRUD ==========
def create_experiment(subject_id: int, condition: str,
                      date: Optional[str] = None, notes: Optional[str] = None) -> Dict[str, Any]:
    """创建实验记录"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO experiments (subject_id, condition, date, notes) VALUES (?, ?, ?, ?)",
        (subject_id, condition, date or datetime.now().strftime('%Y-%m-%d'), notes)
    )
    conn.commit()
    exp_id = cursor.lastrowid
    exp = get_experiment(exp_id)
    conn.close()
    return exp


def get_experiment(exp_id: int) -> Optional[Dict[str, Any]]:
    """获取单个实验记录"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT e.*, s.code as subject_code
           FROM experiments e
           JOIN subjects s ON e.subject_id = s.id
           WHERE e.id = ?""",
        (exp_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_experiments(subject_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """列出实验记录（可按被试筛选）"""
    conn = get_connection()
    cursor = conn.cursor()
    if subject_id:
        cursor.execute(
            """SELECT e.*, s.code as subject_code
               FROM experiments e
               JOIN subjects s ON e.subject_id = s.id
               WHERE e.subject_id = ?
               ORDER BY e.created_at DESC""",
            (subject_id,)
        )
    else:
        cursor.execute(
            """SELECT e.*, s.code as subject_code
               FROM experiments e
               JOIN subjects s ON e.subject_id = s.id
               ORDER BY e.created_at DESC"""
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_experiment(exp_id: int) -> bool:
    """删除实验记录"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# 启动时自动初始化
init_db()
