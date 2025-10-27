from sqlalchemy import create_engine, text
import pandas as pd
import os

DB_PATH = "data/attendance.db"

def get_engine():
    os.makedirs("data", exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")

def init_db():
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            department TEXT
        );
        """))
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ts TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """))

def add_employee(name, department=None):
    eng = get_engine()
    with eng.begin() as con:
        con.execute(
            text("INSERT OR IGNORE INTO employees(name, department) VALUES(:n, :d)"),
            {"n": name, "d": department},
        )

def mark_attendance(name, ts, status="present"):
    eng = get_engine()
    with eng.begin() as con:
        con.execute(
            text("INSERT INTO attendance(name, ts, status) VALUES(:n, :t, :s)"),
            {"n": name, "t": ts, "s": status},
        )

def read_table(table):
    eng = get_engine()
    return pd.read_sql_table(table, eng)

def update_employee(name, **fields):
    if not fields:
        return
    eng = get_engine()
    sets = ", ".join([f"{k}=:{k}" for k in fields.keys()])
    q = text(f"UPDATE employees SET {sets} WHERE name=:name")
    params = {"name": name, **fields}
    with eng.begin() as con:
        con.execute(q, params)

if __name__ == "__main__":
    init_db()
    print("DB ready:", DB_PATH)
