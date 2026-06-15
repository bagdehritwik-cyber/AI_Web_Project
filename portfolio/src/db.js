const fs = require('fs');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();

const dbPath = process.env.DB_PATH || path.join(__dirname, '..', 'data', 'portfolio.sqlite');
fs.mkdirSync(path.dirname(dbPath), { recursive: true });

const db = new sqlite3.Database(dbPath);

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS contact_requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      email TEXT NOT NULL,
      message TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);
});

const run = (query, params = []) =>
  new Promise((resolve, reject) => {
    db.run(query, params, function onRun(err) {
      if (err) return reject(err);
      return resolve({ lastID: this.lastID, changes: this.changes });
    });
  });

const all = (query, params = []) =>
  new Promise((resolve, reject) => {
    db.all(query, params, (err, rows) => {
      if (err) return reject(err);
      return resolve(rows);
    });
  });

module.exports = {
  insertContact: (name, email, message) =>
    run('INSERT INTO contact_requests (name, email, message) VALUES (?, ?, ?)', [name, email, message]),
  getContacts: () =>
    all('SELECT id, name, email, message, created_at FROM contact_requests ORDER BY datetime(created_at) DESC'),
};
