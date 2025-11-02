import sqlite3
from datetime import datetime, date
import os

DB_PATH = "data/attendance.db"

def init_db():
    """Create attendance table if it doesn’t exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT DEFAULT 'present',
            date TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("🗄️ Database initialized at:", DB_PATH)


def mark_attendance(name, status="present"):
    """Insert or update attendance for a person."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = datetime.now().isoformat(timespec='seconds')
    today = date.today().isoformat()

    # Avoid marking same person twice on same day
    cur.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
    existing = cur.fetchone()

    if existing:
        print(f"⚠️ {name} already marked today.")
    else:
        cur.execute("INSERT INTO attendance (name, timestamp, status, date) VALUES (?, ?, ?, ?)",
                    (name, ts, status, today))
        conn.commit()
        print(f"📋 Marked attendance for {name} at {ts}")

    conn.close()


def fetch_today():
    """Fetch today’s attendance list."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    today = date.today().isoformat()
    cur.execute("SELECT name, timestamp, status FROM attendance WHERE date=?", (today,))
    rows = cur.fetchall()
    conn.close()
    return rows
