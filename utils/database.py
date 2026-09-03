import os
import io
import re
import psycopg2
from psycopg2 import sql
import pandas as pd


class DatabaseUtil:
    """
    Thin wrapper around a hosted Postgres connection (Neon / Supabase / RDS /
    Cloud SQL / etc). Different users or uploaded datasets are isolated by
    SCHEMA rather than by separate databases, so this works fine against a
    single hosted instance shared across web requests.
    """

    def __init__(self, db_config: dict):
        self.db_config = db_config
        try:
            self.connection = psycopg2.connect(**db_config)
        except Exception as e:
            # Fail loudly instead of leaving self.connection = None and
            # blowing up later inside a bare `cursor()` call.
            raise ConnectionError(f"Error connecting to the database: {e}")

    def close(self):
        if self.connection and not self.connection.closed:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------ schema mgmt

    def create_schema(self, schema_name: str):
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
            )
        self.connection.commit()

    def list_schemas(self) -> list:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('pg_catalog','information_schema') "
                "AND schema_name NOT LIKE 'pg_toast%' AND schema_name NOT LIKE 'pg_temp%';"
            )
            return [row[0] for row in cursor.fetchall()]

    def drop_schema(self, schema_name: str, cascade: bool = True):
        """Use this to tear down a user's sandbox schema when their session ends."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} {}").format(
                    sql.Identifier(schema_name),
                    sql.SQL("CASCADE" if cascade else ""),
                )
            )
        self.connection.commit()

    # ------------------------------------------------------- dynamic tables

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        name = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_").lower()
        if not name:
            name = "col"
        if name[0].isdigit():
            name = f"_{name}"
        return name

    @staticmethod
    def _infer_pg_type(dtype) -> str:
        dtype_str = str(dtype)
        if "int" in dtype_str:
            return "BIGINT"
        if "float" in dtype_str:
            return "DOUBLE PRECISION"
        if "bool" in dtype_str:
            return "BOOLEAN"
        if "datetime" in dtype_str:
            return "TIMESTAMP"
        return "TEXT"

    def create_table_from_dataframe(
        self, df: pd.DataFrame, schema_name: str, table_name: str, drop_if_exists: bool = True
    ):
        table_name = self._sanitize_identifier(table_name)
        columns = [(self._sanitize_identifier(c), self._infer_pg_type(t)) for c, t in df.dtypes.items()]

        with self.connection.cursor() as cursor:
            if drop_if_exists:
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier(schema_name), sql.Identifier(table_name)
                    )
                )
            col_defs = sql.SQL(", ").join(
                sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(t)) for c, t in columns
            )
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name), col_defs
                )
            )
        self.connection.commit()
        return table_name, [c for c, _ in columns]

    def load_dataframe(self, df: pd.DataFrame, schema_name: str, table_name: str):
        table_name = self._sanitize_identifier(table_name)
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)

        columns = [self._sanitize_identifier(c) for c in df.columns]

        copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT CSV, NULL '')").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        )
        with self.connection.cursor() as cursor:
            cursor.copy_expert(copy_sql, buf)
        self.connection.commit()

    def load_csv_file(
        self, csv_path: str, schema_name: str, table_name: str = None, drop_if_exists: bool = True
    ) -> str:
        """
        Infers a table schema straight from a CSV's pandas dtypes, creates the
        table (in `schema_name`), and loads the data. This is what replaces
        the old hardcoded CREATE TABLE statements in feed_db.py — it works
        for *any* CSV a user uploads, not just the 5 known ones.
        """
        table_name = table_name or os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        self.create_table_from_dataframe(df, schema_name, table_name, drop_if_exists=drop_if_exists)
        self.load_dataframe(df, schema_name, table_name)
        return self._sanitize_identifier(table_name)

    # -------------------------------------------------------- inspection / query

    def schema_details(self, schema_name: str) -> str:
        schema_info_context = f"Database Schema: {schema_name}\n"
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema_name,),
            )
            tables_list = cursor.fetchall()

            for (table_name,) in tables_list:
                schema_info_context += f"\nTable: {table_name}\n"

                # NOTE: original code filtered columns only by table_name, which
                # could bleed columns in from a same-named table in another
                # schema. Filtering by table_schema too, since we now have many.
                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s;",
                    (schema_name, table_name),
                )
                for column_name, data_type in cursor.fetchall():
                    schema_info_context += f"  Column: {column_name}, Data Type: {data_type}\n"

                cursor.execute(
                    sql.SQL("SELECT * FROM {}.{} LIMIT 5;").format(
                        sql.Identifier(schema_name), sql.Identifier(table_name)
                    )
                )
                schema_info_context += "  Sample Data:\n"
                for row in cursor.fetchall():
                    schema_info_context += f"    {row}\n"

        return schema_info_context

    def execute_sql(self, query: str):
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall() if cursor.description else []
            self.connection.commit()
            return str(result)

    def execute_query_df(self, query: str) -> pd.DataFrame:
        """Runs a SELECT and returns a DataFrame — used for CSV downloads."""
        return pd.read_sql(query, self.connection)