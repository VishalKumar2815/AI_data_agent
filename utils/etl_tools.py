import os
import requests
import pandas as pd
import builtins

from utils.database import DatabaseUtil
from utils.llm_pick import pick_llm


class ETLTools:

    def __init__(self):
        pass

    # ------------------------------------------------------------- extract (API -> file)

    def extract_load(self,url:str, output_folder:str, format:str):
        """
        This tool extracts the data from the API (url) and loads it into the
        the desired location (output_folder).

        Args:
            url (str): The API endpoint from which to extract data.
            output_folder (str): The folder where the extracted data will be saved.
        
        Returns:
            str: A message indicating the success or failure of the operation.

        """

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder = os.path.join(project_root, output_folder)      

        try:
            response = requests.get(url)
            response.raise_for_status()
            data  = response.json()

            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            df = pd.json_normalize(data['results'])
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}"

            return f"Data successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"


    def transform_load_context(self, file_path:str):
        """
        This tool transforms the data from the specified file and loads it into the
        desired location (output_folder).

        Args:
            file_path (str): The path to the file containing the data to be transformed.
            output_folder (str): The folder where the transformed data will be saved.
            output_format (str): The format in which to save the transformed data (csv, json, parquet).
        Returns:
            str: A message indicating the success or failure of the operation.
        """

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}"

        top_3_rows = str(df.head(3))

        return top_3_rows


    def execute_code(self,code:str):
        """
        This tool executes the provided code and returns the output.

        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        """

        try:
            exec(code)
            return "Code executed successfully."
        except Exception as e:
            return f"Failed to execute code: {e}"

    # ------------------------------------------------------------- transform (DB table -> DB table)
    # Used by the chat 'transform' route: operates on tables already loaded into
    # Postgres (uploaded/extracted), rather than local files.

    def list_tables(self, conn_details: dict, schema_name: str) -> list:
        db = DatabaseUtil(conn_details)
        try:
            with db.connection.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                    (schema_name,),
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            db.close()

    def pick_target_table(self, conn_details: dict, schema_name: str, user_question: str) -> str:
        """If only one table is loaded, use it. Otherwise, ask the LLM which
        table the user's request refers to."""
        tables = self.list_tables(conn_details, schema_name)
        if not tables:
            raise ValueError(f"No tables found in schema {schema_name}.")
        if len(tables) == 1:
            return tables[0]

        llm = pick_llm("low")
        prompt = f"""
        The user wants to transform one of these tables: {tables}
        Based on their request below, reply with ONLY the exact table name
        from the list above — nothing else, no explanation.

        Request: {user_question}
        """
        choice = llm.invoke(prompt).content.strip().strip("`").strip()
        return choice if choice in tables else tables[0]

    def apply_transformation(self, conn_details: dict, schema_name: str, table_name: str, user_question: str) -> dict:
        """
        Fetches the table, has the LLM write pandas code for the requested
        transformation, executes it in a restricted namespace (no builtins,
        no file/network access), and saves the result as a new table
        `<table_name>_transformed` in the same schema.

        Returns dict with keys: message, target_table, row_count, code
        """
        db = DatabaseUtil(conn_details)
        try:
            df = db.execute_query_df(f'SELECT * FROM "{schema_name}"."{table_name}"')

            llm = pick_llm("low")
            prompt = f"""
            You are a Python data analyst. A pandas DataFrame called `df` is
            already loaded with this data:

            Columns and types:
            {df.dtypes.to_string()}

            Sample rows:
            {df.head(3).to_string()}

            User's request: {user_question}

            Write ONLY pandas code (no explanation, no markdown fences, no
            imports, no file or network access) that transforms `df`
            according to the request and assigns the final result to a
            variable named `result_df`.
            """

            code = llm.invoke(prompt).content.strip()
            code = code.strip("`")
            if code.lower().startswith("python"):
                code = code[6:].strip()

            # Restricted namespace: only a safe whitelist of builtins is
            # available (no open/import/exec/eval/__import__), plus pandas
            # and the input dataframe.
            allowed_builtin_names = [
                "len", "range", "list", "dict", "set", "tuple", "str", "int", "float",
                "bool", "sum", "min", "max", "sorted", "reversed", "enumerate", "zip",
                "map", "filter", "abs", "round", "any", "all", "isinstance", "print",
            ]
            safe_builtins = {name: getattr(builtins, name) for name in allowed_builtin_names}
            safe_globals = {"__builtins__": safe_builtins, "pd": pd}
            local_vars = {"df": df.copy()}

            try:
                exec(code, safe_globals, local_vars)
            except Exception as e:
                return {
                    "message": f"Failed to execute transformation: {e}",
                    "target_table": None,
                    "row_count": 0,
                    "code": code,
                }

            result_df = local_vars.get("result_df")
            if result_df is None or not isinstance(result_df, pd.DataFrame):
                return {
                    "message": "Transformation code did not produce a valid result_df.",
                    "target_table": None,
                    "row_count": 0,
                    "code": code,
                }

            target_table = f"{table_name}_transformed"
            db.create_table_from_dataframe(result_df, schema_name, target_table, drop_if_exists=True)
            db.load_dataframe(result_df, schema_name, target_table)

            return {
                "message": f"Transformed '{table_name}' ({len(df)} rows) into '{target_table}' ({len(result_df)} rows).",
                "target_table": target_table,
                "row_count": len(result_df),
                "code": code,
            }
        finally:
            db.close()


if __name__ == "__main__":
    obj = ETLTools()
    path = "C:\\Data_Agent\\data\\extract\\extracted_data.csv"
    print(obj.transform_load_context(path))