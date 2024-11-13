from sshtunnel import SSHTunnelForwarder
import snowflake.connector
import pymysql
from db_utils.os_utils import get_project_data_path
from config.settings import (
    empire_analytics_db,
    empire_backstage_db,
    empire_snowflake_db_staging,
)
import pandas as pd

# import paramiko
import time
import io
from warnings import filterwarnings
from cryptography.utils import CryptographyDeprecationWarning

filterwarnings("ignore", category=CryptographyDeprecationWarning)
filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*pandas only supports SQLAlchemy connectable.*",
)


def start_empire_tunnel(db):
    key = io.StringIO(empire_analytics_db.pem_key)
    # mypkey = paramiko.RSAKey.from_private_key(key, password="password")

    if db == "analytics":
        host = empire_analytics_db.host
        port = empire_analytics_db.sql_port
    elif db == "backstage":
        host = empire_backstage_db.host
        port = empire_backstage_db.sql_port
    tunnel = SSHTunnelForwarder(
        (empire_analytics_db.ssh_host, empire_analytics_db.ssh_port),
        ssh_username=empire_analytics_db.ssh_user,
        # ssh_pkey=mypkey,
        ssh_password="ssh_password",
        remote_bind_address=(host, port),
    )
    tunnel.start()
    return tunnel


def get_analytics_conn(tunnel):
    conn = pymysql.connect(
        host="localhost",
        user=empire_analytics_db.sql_username,
        passwd=empire_analytics_db.sql_password,
        db=empire_analytics_db.default_db,
        port=tunnel.local_bind_port,
        ssl={"fake_flag_to_enable_tls": True},
        charset="utf8",
        use_unicode=True,
    )
    return conn


def get_backstage_conn(tunnel):
    conn = pymysql.connect(
        host="localhost",
        user=empire_backstage_db.sql_username,
        passwd=empire_backstage_db.sql_password,
        db=empire_backstage_db.default_db,
        port=tunnel.local_bind_port,
        ssl={"fake_flag_to_enable_tls": True},
    )
    return conn


def get_snowflake_conn():
    conn = snowflake.connector.connect(
        user=empire_snowflake_db_staging.user,
        password=empire_snowflake_db_staging.passw,
        account=empire_snowflake_db_staging.account,
        warehouse=empire_snowflake_db_staging.warehouse,
        database=empire_snowflake_db_staging.database,
        schema=empire_snowflake_db_staging.schema,
    )
    return conn


def get_query(db, name, query):
    data = get_project_data_path()
    try:
        df = pd.read_csv(f"{data}/{db}_{name}.csv", encoding="UTF-8")
        print(f"File {db}_{name}.csv loaded successfully")
    except:
        print(f"Query from {db}, {name} has been started")
        if db == "analytics":
            tunnel = start_empire_tunnel("analytics")
            conn = get_analytics_conn(tunnel)
        elif db == "backstage":
            tunnel = start_empire_tunnel("backstage")
            conn = get_backstage_conn(tunnel)
        elif db == "snowflake":
            conn = get_snowflake_conn()
        start = time.time()
        df = pd.read_sql(query, conn, coerce_float=False)
        end = time.time()
        print(f"Query from {db}, {name} took {end - start} seconds")
        df.to_csv(f"{data}/{db}_{name}.csv", index=False)
    return df


if __name__ == "__main__":
    query = """WITH spotify_streams_data AS (
                SELECT 
                    DATE_TRUNC(day, sa.date) AS streams_date, 
                    sa.isrc, 
                    SUM(streams_total) AS spotify_streams
                FROM spotify_streams_aggregated sa
                GROUP BY DATE_TRUNC(day, sa.date), sa.isrc
            ),
            artists AS (
                SELECT 
                    apple_identifier, 
                    isrc, 
                    artist, 
                    artist_id,
                    ROW_NUMBER() OVER (PARTITION BY isrc ORDER BY date_statement DESC) AS rn
                FROM applemusic_content
            ),
            explicit_info AS (
                SELECT 
                    isrc, 
                    explicit, 
                    explicit_text,
                    ROW_NUMBER() OVER (PARTITION BY isrc ORDER BY date_statement DESC) AS rn
                FROM analytics_detailed_spotify
            ),
            spotify_detailed_playlist_data AS (
                SELECT 
                    DATE_TRUNC(day, sa.date) AS streams_date, 
                    sa.isrc, 
                    SUM(streams_total) AS streams_playlist, 
                    sa.followers, 
                    sa.playlist_id
                FROM PROD_DWH.DWH.ANALYTICS_DETAILED_SPOTIFY_PLAYLISTS sa
                GROUP BY DATE_TRUNC(day, sa.date), sa.isrc, sa.playlist_id, sa.followers
            ),
            earliest_album_per_isrc AS (
                -- Subquery to get the earliest release date for each ISRC
                SELECT 
                    isrc,
                    MIN(release_date) AS earliest_release_date
                FROM prod_dwh.staging.uma_albums_source a
                JOIN prod_dwh.staging.uma_tracks_source t ON t.album_id = a.album_id
                GROUP BY isrc
            ),
            spotify_history_data AS (
                SELECT 
                    th.info, 
                    th.timestamp, 
                    tr.isrc, 
                    ts.playlist_id,
                    tr.genre_id, 
                    ea.earliest_release_date as release_date,
                    tr.label_id,
                    al.explicit_lyrics
                FROM SPOTIFY_PLAYLIST_TRACK_HISTORY_SOURCE th
                JOIN SPOTIFY_PLAYLIST_TRACKS_SOURCE ts ON ts.ID = th.spotify_playlist_track_id
                JOIN prod_dwh.staging.uma_tracks_source tr ON tr.track_id = ts.track_id
                JOIN earliest_album_per_isrc ea ON ea.isrc = tr.isrc  -- Join to get the earliest album for each ISRC
                JOIN prod_dwh.staging.uma_albums_source al ON al.album_id = tr.album_id    
                WHERE ts.playlist_id IN ('37i9dQZF1DX0XUsuxWHRQd', '37i9dQZF1DWY4xHQp97fN6', '6UeSakyzhiEt4NB3UAd6NQ', '37i9dQZF1DZ06evO152G0U', '37i9dQZF1DZ06evO3oLcOc', '37i9dQZF1DX48TTZL62Yht', '37i9dQZF1DX97h7ftpNSYT', '37i9dQZF1DWXT8uSSn6PRy', '37i9dQZEVXbKCF6dqVpDkS', '2NoR0KhNZ8oZnC3HoNt2FV', '6fi0ExpvtWx0vQUelJkwmV') 
                  --AND ea.earliest_release_date < '2024-05-01' 
                  --AND ea.earliest_release_date > '2021-08-01'
                  and (tr.isrc = 'USAT22305462' or tr.isrc = 'USUYG1383753')
            ),
            isrc_occurrences AS (
                -- Count the number of occurrences of "Added to playlist at position XX" for each ISRC
                SELECT 
                    isrc, 
                    COUNT(*) AS add_count
                FROM spotify_history_data
                WHERE info LIKE 'Added to playlist at position%'
                GROUP BY isrc
            ),
            filtered_isrcs AS (
                -- Filter to get only those ISRCs where the add_count is less than 3
                SELECT 
                    isrc
                FROM isrc_occurrences
                WHERE add_count > 0 AND add_count < 3
            ),
            playlist_data_transformed AS (
                SELECT 
                    th.info, 
                    DATE_TRUNC(day, th.timestamp) AS timestamp, 
                    ts.playlist_id, 
                    ts.track_id,
                    -- Transform the position data
                    CASE 
                        WHEN th.info LIKE 'Added to playlist at position%' THEN CAST(IFNULL(REGEXP_SUBSTR(th.info, '[0-9]+$'), 0) AS INTEGER)
                        WHEN th.info LIKE 'Position changed from%' THEN CAST(IFNULL(REGEXP_SUBSTR(th.info, '[0-9]+$'), 0) AS INTEGER)
                        WHEN th.info = 'Removed from playlist' THEN CAST(0 AS INTEGER)
                        ELSE CAST(NULL AS INTEGER)
                    END AS position,
                    -- Add event column based on the type of event
                    CASE 
                        WHEN th.info LIKE 'Added to playlist%' THEN 1
                        WHEN th.info = 'Removed from playlist' THEN 2
                        ELSE 0
                    END AS event,
                    CASE
                        WHEN featured_artist IS NOT NULL AND featured_artist != '' THEN 1
                        ELSE 0
                    END AS featured_artist,
                    tr.isrc
                FROM SPOTIFY_PLAYLIST_TRACK_HISTORY_SOURCE th
                JOIN SPOTIFY_PLAYLIST_TRACKS_SOURCE ts ON ts.ID = th.spotify_playlist_track_id
                JOIN prod_dwh.staging.uma_tracks_source tr ON tr.track_id = ts.track_id
            )
            -- Now select all the data for the filtered ISRCs
            SELECT 
                shd.info, 
                pdt.position,
            FROM spotify_history_data shd
            JOIN filtered_isrcs fi ON shd.isrc = fi.isrc  -- Join to keep only ISRCs with add_count < 3
            JOIN spotify_detailed_playlist_data sp ON sp.isrc = shd.isrc AND sp.streams_date = DATE_TRUNC(day, shd.timestamp) AND sp.playlist_id = shd.playlist_id
            JOIN explicit_info ads ON ads.isrc = fi.isrc AND ads.rn = 1
            JOIN artists a ON a.isrc = shd.isrc AND a.rn = 1
            LEFT JOIN spotify_streams_data st ON st.isrc = shd.isrc AND st.streams_date = DATE_TRUNC(day, shd.timestamp)
            LEFT JOIN playlist_data_transformed pdt ON pdt.playlist_id = shd.playlist_id AND DATE_TRUNC(day, pdt.timestamp) = DATE_TRUNC(day, shd.timestamp) AND pdt.isrc = shd.isrc
            ORDER BY shd.isrc, shd.timestamp;"""
    df = get_query("snowflake", "test", query)
    print(df)
