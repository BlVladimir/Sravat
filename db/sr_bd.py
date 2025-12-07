import sqlite3

connect = sqlite3.connect('../Sravat.db', check_same_thread=False)
cursor = connect.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS files (
    id TEXT,
    file TEXT
)
''')
def plus_file(id, file):
    cursor.execute('''
    INSERT INTO 'files' id=?, file=?''', (id, file))
