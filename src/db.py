"""
DWDM Project: MySQL Database Connection & Query Utility
Auto-converts Decimal and date objects for seamless JSON serialization.
"""

import datetime
import decimal
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import pymysql
from pymysql.cursors import DictCursor

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "bmc_complaint_dw")
DB_PORT = int(os.getenv("DB_PORT", 3306))


def serialize_cell(val):
    if isinstance(val, decimal.Decimal):
        return float(val)
    elif isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    return val


def serialize_row(row):
    if row is None:
        return None
    return {k: serialize_cell(v) for k, v in row.items()}


@contextmanager
def get_db():
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        cursorclass=DictCursor,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()


def execute_query(sql, params=None):
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            rows = cursor.fetchall()
            return [serialize_row(r) for r in rows]


def execute_one(sql, params=None):
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            row = cursor.fetchone()
            return serialize_row(row)
