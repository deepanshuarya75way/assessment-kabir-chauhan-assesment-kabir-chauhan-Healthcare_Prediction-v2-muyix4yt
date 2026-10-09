import sqlite3
from pathlib import Path

base_dir = Path(__file__).resolve().parent
db_path =  base_dir / "healthcare.db"

def get_connection():
  """Create a connection to the SQLite database."""
  connection=  sqlite3.connect(db_path)
  connection.row_factory = sqlite3.Row
  return connection

def init_db():
  """Create the report if it doesnt exist"""
  with get_connection() as connection:
    connection.execute("""
      CREATE TABLE IF NOT EXISTS prediction_reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease TEXT NOT NULL CHECK(disease IN ('diabetes', 'heart')),
        patient_inputs TEXT NOT NULL,
        prediction_result TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING'
          CHECK (
            status IN (
              'PENDING',
              'APPROVED',
              'REJECTED'
            )
          ),
        created_at TEXT NOT NULL,
        email_sent_at TEXT,
        expires_at TEXT,
        action_token_hash TEXT UNIQUE,
        decision_at TEXT
      )
    """)
    connection.commit()

print("DB working.")
print(f"Database Location: {db_path}")

if __name__ == "__main__":
  init_db()
  with get_connection() as connection:
    columns = connection.execute("PRAGMA table_info(prediction_reports)").fetchall()
    existing_columns = [row["name"] for row in columns]
    if "prediction_result" not in existing_columns:
      connection.execute(
        "ALTER TABLE prediction_reports "
        "ADD COLUMN prediction_result TEXT"
      )
      connection.commit()
      print("Add prediction_result column.")
    else:
      print("Already there")

