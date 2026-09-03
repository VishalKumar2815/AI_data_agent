import os
from utils.database import DatabaseUtil


def get_conn_details() -> dict:
    """
    Connection details for the hosted Postgres instance (Neon / Supabase /
    RDS / Cloud SQL / etc). One instance is shared across all web-server
    users; isolation between users happens at the schema level, not the
    connection level.

    Set DATABASE_URL for the common hosted-Postgres case, or fall back to
    discrete host/port/user/password/database env vars.
    """
    if "DATABASE_URL" in os.environ:
        return {"dsn": os.environ["DATABASE_URL"], "sslmode": os.environ.get("sslmode", "require")}

    return {
        "host": os.environ["host"],
        "port": int(os.environ.get("port", 5432)),
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["database"],
        "sslmode": os.environ.get("sslmode", "require"),
    }


def schema_name_for(user_id: str) -> str:
    """Deterministic, safe schema name per user/session, e.g. 'user_7f3ac1'."""
    safe = "".join(c for c in str(user_id) if c.isalnum() or c == "_").lower()
    return f"user_{safe or 'anon'}"


def load_user_csvs(csv_paths: list, schema_name: str, drop_if_exists: bool = True) -> dict:
    """
    Loads one or more user-uploaded CSVs into their own isolated Postgres
    schema. Each CSV becomes one table (named after the file), with columns
    and types inferred straight from the data — no hardcoded table
    definitions, so it works for whatever the user uploads.

    Call this from the web server's upload endpoint, once per
    user/session, before invoking the agent.

    Args:
        csv_paths: local paths to the uploaded CSV files (e.g. saved from
            an upload handler to a temp dir).
        schema_name: isolated schema for this user/session. Use
            schema_name_for(user_id) to generate one.
        drop_if_exists: replace any existing table of the same name
            (useful when a user re-uploads a corrected file).

    Returns:
        dict mapping table_name -> row count loaded.
    """
    conn_details = get_conn_details()
    loaded = {}

    with DatabaseUtil(conn_details) as db:
        db.create_schema(schema_name)
        for path in csv_paths:
            table_name = db.load_csv_file(path, schema_name, drop_if_exists=drop_if_exists)
            with db.connection.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}";')
                loaded[table_name] = cur.fetchone()[0]

    return loaded


def cleanup_user_schema(schema_name: str):
    """Call when a user's session/upload should be discarded."""
    conn_details = get_conn_details()
    with DatabaseUtil(conn_details) as db:
        db.drop_schema(schema_name, cascade=True)


# if __name__ == "__main__":
#     # Example: a web request just saved these to a temp upload dir
#     uploaded = ["data/users.csv", "data/rides.csv"]
#     schema = schema_name_for(user_id="demo-user-123")
#     print(load_user_csvs(uploaded, schema))