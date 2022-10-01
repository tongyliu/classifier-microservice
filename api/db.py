import os
import pathlib

import MySQLdb


def _connect():
    """ https://mysqlclient.readthedocs.io/user_guide.html#mysqldb """
    return MySQLdb.connect(host=os.environ['MYSQL_HOST'],
                           user=os.environ['MYSQL_USER'],
                           passwd=os.environ['MYSQL_PASSWORD'],
                           db=os.environ['MYSQL_DATABASE'])


class DatabaseManager:
    def __init__(self):
        self._conn = _connect()
        self._load_queries()
    
    def setup_tables(self, drop_existing: bool = False) -> None:
        cursor = self._conn.cursor()

        if drop_existing:
            cursor.execute(self._queries['drop_tables'])
        
        cursor.execute(self._queries['create_tables'])
        
    def _load_queries(self):
        # Read queries into memory so we don't need to repeat filesystem access
        self._queries = {}
        query_dir = pathlib.Path("./sql")

        for query_file in query_dir.iterdir():
            with open(query_file) as f:
                query_name = query_file.stem
                self._queries[query_name] = f.read()
