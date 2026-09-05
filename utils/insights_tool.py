import pandas as pd
from utils.database import DatabaseUtil


class InsightsTools:

    def __init__(self, conn_details: dict):
        self.conn_details = conn_details

    def _table_summary(self, db: DatabaseUtil, schema_name: str, table_name: str) -> str:
        df = db.execute_query_df(f'SELECT * FROM "{schema_name}"."{table_name}"')

        summary = f"Table: {table_name}\n"
        summary += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
        summary += f"Columns & types:\n{df.dtypes.to_string()}\n"

        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            summary += f"Null counts (columns with missing values):\n{nulls.to_string()}\n"

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            summary += f"Numeric summary:\n{numeric_df.describe().to_string()}\n"

        categorical_df = df.select_dtypes(include="object")
        if not categorical_df.empty:
            summary += "Top values per categorical column:\n"
            for col in categorical_df.columns[:10]:  # cap columns to avoid blowing up the prompt
                top_vals = categorical_df[col].value_counts().head(5)
                summary += f"  {col}: {top_vals.to_dict()}\n"

        summary += f"Sample rows:\n{df.head(3).to_string()}\n"
        return summary

    def build_data_summary(self, schema_name: str, table_name: str = None) -> str:
        """
        Summarizes either one named table or every table in the schema —
        stats only, never the raw dataset — so the LLM has enough context to
        reason about the data without us shipping the whole table to it.
        """
        db = DatabaseUtil(self.conn_details)
        try:
            if table_name:
                tables = [DatabaseUtil._sanitize_identifier(table_name)]
            else:
                with db.connection.cursor() as cur:
                    cur.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                        (schema_name,),
                    )
                    tables = [row[0] for row in cur.fetchall()]

            if not tables:
                return f"No tables found in schema {schema_name}."

            summaries = [self._table_summary(db, schema_name, t) for t in tables]
            return "\n\n".join(summaries)
        finally:
            db.close()