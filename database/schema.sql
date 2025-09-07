CREATE TABLE cases (
  case_id TEXT PRIMARY KEY,
  task TEXT,
  tags TEXT,
  solution TEXT,
  embedding BYTEA,
  q_value REAL DEFAULT 0,
  reward REAL DEFAULT 0,
  reward_count INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);