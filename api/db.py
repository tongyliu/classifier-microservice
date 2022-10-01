import base64
import os
import pathlib
from typing import Any, Dict, List, Optional

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

    def create_model(
        self,
        model: str,
        params: str,
        d: int,
        n_classes: int,
        model_pkl: bytes,
    ) -> int:
        model_pkl = base64.b64encode(model_pkl)

        cursor = self._conn.cursor()
        cursor.execute(self._queries['create_model'], (model, params, d, n_classes, model_pkl))
        cursor.execute(self._queries['get_insert_id'])
        new_id = cursor.fetchone()[0]
        cursor.close()
        self._conn.commit()

        return int(new_id)

    def get_model(self, model_id: int) -> Optional[Dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute(self._queries['get_model'], (model_id,))
        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None

        values_dict = {k[0]: v for k, v in zip(cursor.description, row)}
        values_dict["model_pkl"] = base64.b64decode(values_dict["model_pkl"])

        return values_dict

    def update_model(self, model_id: int, model_pkl: bytes, n_trained: int) -> None:
        model_pkl = base64.b64encode(model_pkl)

        cursor = self._conn.cursor()
        cursor.execute(self._queries['update_model'], (model_pkl, n_trained, model_id))
        cursor.close()
        self._conn.commit()

    def get_models(self) -> List[Dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute(self._queries['get_models'])
        rows = cursor.fetchall()
        cursor.close()
        
        models = [
            {k[0]: v for k, v in zip(cursor.description, row)}
            for row in rows
        ]

        return models

    def _load_queries(self):
        # Read queries into memory so we don't need to repeat filesystem access
        self._queries = {}
        query_dir = pathlib.Path('./sql')

        for query_file in query_dir.iterdir():
            with open(query_file) as f:
                query_name = query_file.stem
                self._queries[query_name] = f.read()
