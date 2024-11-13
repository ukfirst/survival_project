import numpy as np

from db_utils.db_utils import get_query
import pandas as pd

from dataclasses import dataclass


@dataclass
class EdaMusicDataModel:
    """Class to gather different music data for analysis
    Пример:
    data = MusicService().get_music_data(track_amount=50, track_offset=10, min_playlist_followers=5000000)
    """

    tracks: pd.DataFrame = None
    artists: pd.DataFrame = None
    genres: pd.DataFrame = None
    labels: pd.DataFrame = None
    playlists: pd.DataFrame = None
    discovery: pd.DataFrame = None


class MusicService:
    def __init__(self):
        self._music_data = None
        self._merged_data = None

    def get_artist_list(self, df):
        """Get list of artist from dataframe"""
        # artists = df["artist_id"].unique()
        # count = len(artists)
        # artists = repr(tuple(map(str, artists)))
        # if artists[-2] == ",":
        #     isrc = artists.replace(artists[-2], "")
        """Get list of unique artist IDs from dataframe with concatenated artist IDs."""
        # Flattening and splitting the artist_ids
        artist_ids = df["artist_id"].astype(str).str.split(",")

        # Flattening the list of lists and converting it to unique set
        artists_flat = set(
            artist for artist_list in artist_ids for artist in artist_list
        )

        # Convert set to sorted list for consistent ordering
        artists_flat = sorted(artists_flat)

        # Count of unique artists
        count = len(artists_flat)

        # Formatting the tuple representation of artist IDs
        artists_repr = repr(tuple(artists_flat))

        # Handling the trailing comma for single item tuples
        if len(artists_flat) == 1:
            artists_repr = artists_repr[:-2] + ")"
        return artists_repr, count

    def get_genre_list(self, df):
        """Get list of genres from dataframe"""
        genres = df["genre_id"].unique()
        count = len(genres)
        # Format genres based on the count of unique items
        if count == 1:
            genres_str = f"('{genres[0]}')"  # Single genre without trailing comma
        else:
            genres_str = repr(
                tuple(map(str, genres))
            )  # Multiple genres in tuple format
        return genres_str, count

    def get_isrc_list(self, df):
        isrcs = df["isrc"].unique()
        count = len(isrcs)
        # Format genres based on the count of unique items
        if count == 1:
            isrcs_str = f"('{isrcs[0]}')"  # Single genre without trailing comma
        else:
            isrcs_str = repr(tuple(map(str, isrcs)))  # Multiple genres in tuple format
        return isrcs_str, count

    def get_playlist_list(self, df):
        """Get list of playlists from dataframe"""
        playlists = df["playlist_id"].unique()
        count = len(playlists)
        # Format genres based on the count of unique items
        if count == 1:
            playlists_str = f"('{playlists[0]}')"  # Single genre without trailing comma
        else:
            playlists_str = repr(
                tuple(map(str, playlists))
            )  # Multiple genres in tuple format
        return playlists_str, count

    def get_label_list(self, df):
        """Get list of genres from dataframe"""
        labels = df["label_id"].unique()
        count = len(labels)
        # Format genres based on the count of unique items
        if count == 1:
            labels_str = f"('{labels[0]}')"  # Single genre without trailing comma
        else:
            labels_str = repr(
                tuple(map(str, labels))
            )  # Multiple genres in tuple format
        return labels_str, count

    def get_tracks_by_popularity(self):
        query = """
                WITH earliest_album_per_isrc AS (
                    -- Subquery to get the earliest release date for each ISRC
                    SELECT 
                        isrc,
                        MIN(release_date) AS earliest_release_date
                    FROM prod_dwh.staging.uma_albums_source a
                    JOIN prod_dwh.staging.uma_tracks_source t ON t.album_id = a.album_id
                    GROUP BY isrc
                ),
                spotify_detailed_playlist_data AS (
                    SELECT 
                        DATE_TRUNC('day', sa.date) AS streams_date, 
                        sa.isrc, 
                        SUM(streams_total) AS streams_playlist, 
                        sa.followers, 
                        sa.playlist_id
                    FROM PROD_DWH.DWH.ANALYTICS_DETAILED_SPOTIFY_PLAYLISTS sa
                    JOIN earliest_album_per_isrc ea ON ea.isrc = sa.isrc
                    WHERE ea.earliest_release_date > '2021-07-01'  and ea.earliest_release_date < '2021-10-01'
                      AND DATE_TRUNC('day', sa.date) > '2021-07-01'
                    GROUP BY DATE_TRUNC('day', sa.date), sa.isrc, sa.playlist_id, sa.followers
                ),
                total_streams_per_isrc AS (
                    -- Calculate total streams for each ISRC, filter by total_streams > 35000
                    SELECT 
                        isrc,
                        SUM(streams_playlist) AS total_streams
                    FROM spotify_detailed_playlist_data
                    GROUP BY isrc
                    HAVING SUM(streams_playlist) > 1500000  -- Filter to include only tracks with total streams > 100000
                ),
                tracks AS (
                    -- Retrieve additional track-level information such as label_id, genre_id, artist_id, album_id
                    SELECT 
                        ut.isrc, 
                        ut.label_id, 
                        ut.genre_id, 
                        ut.album_id, 
                        ut.track_id, 
                        ut.performer,
                        ROW_NUMBER() OVER (PARTITION BY ut.isrc ORDER BY ut.isrc DESC) AS rn
                    FROM prod_dwh.staging.uma_tracks_source ut
                ),
                artists AS (
                    -- Retrieve artist information for each track
                    SELECT 
                        apple_identifier, 
                        isrc, 
                        artist, 
                        artist_id,
                        ROW_NUMBER() OVER (PARTITION BY isrc ORDER BY date_statement DESC) AS rn
                    FROM applemusic_content
                ),
                tracks_with_metadata AS (
                    -- Join tracks and artists to get metadata
                    SELECT 
                        t.isrc,
                        t.label_id,
                        t.genre_id,
                        t.album_id,
                        a.artist_id
                    FROM tracks t
                    LEFT JOIN artists a ON t.isrc = a.isrc AND a.rn = 1
                    WHERE t.rn = 1  -- Ensure one unique record per isrc
                ),
                ranked_tracks AS (
                    -- Assign each track to a percentile based on total streams
                    SELECT 
                        tsp.isrc,
                        tsp.total_streams,
                        twm.label_id,
                        twm.genre_id,
                        twm.album_id,
                        twm.artist_id,
                        NTILE(100) OVER (ORDER BY tsp.total_streams DESC) AS percentile_rank
                    FROM total_streams_per_isrc tsp
                    LEFT JOIN tracks_with_metadata twm ON tsp.isrc = twm.isrc
                ),
                classified_tracks AS (
                    -- Classify tracks based on percentiles
                    SELECT 
                        isrc,
                        total_streams,
                        label_id,
                        genre_id,
                        album_id,
                        artist_id,
                        percentile_rank,
                        CASE 
                            WHEN percentile_rank <= 5 THEN 'star'
                            WHEN percentile_rank <= 30 THEN 'popular'
                            WHEN percentile_rank <= 60 THEN 'middle'
                            ELSE 'low'
                        END AS popularity_category
                    FROM ranked_tracks
                )
                SELECT 
                    isrc,
                    total_streams,
                    label_id,
                    genre_id,
                    album_id,
                    artist_id,
                    percentile_rank,
                    popularity_category
                FROM classified_tracks
                ORDER BY percentile_rank, total_streams DESC;
        """
        track_data = get_query("snowflake", "eda_tracks", query)
        track_data.columns = [x.lower() for x in track_data.columns]
        return track_data

    def get_tracks_in_playlists_data(self, df):
        """
        Get best track data from database with given offset from the top
        """
        isrc, count = self.get_isrc_list(df)
        query = f"""
            WITH earliest_album_per_isrc AS (
                -- Subquery to get the earliest release date for each ISRC
                SELECT 
                    isrc,
                    MIN(release_date) AS earliest_release_date
                FROM prod_dwh.staging.uma_albums_source a
                JOIN prod_dwh.staging.uma_tracks_source t ON t.album_id = a.album_id
                GROUP BY isrc
            ),
            spotify_detailed_playlist_data AS (
                SELECT 
                    DATE_TRUNC('day', sa.date) AS streams_date, 
                    sa.isrc, 
                    SUM(streams_total) AS streams_playlist, 
                    sa.followers, 
                    sa.playlist_id
                FROM PROD_DWH.DWH.ANALYTICS_DETAILED_SPOTIFY_PLAYLISTS sa
                WHERE sa.isrc IN {(isrc)}
                GROUP BY DATE_TRUNC('day', sa.date), sa.isrc, sa.playlist_id, sa.followers
            ),
            playlist_stream_categories AS (
                SELECT
                    streams_date,
                    isrc,
                    streams_playlist,
                    followers,
                    playlist_id,
                    CASE 
                        WHEN followers = 0 AND (playlist_id IS NULL OR playlist_id ='') THEN 'direct'
                        WHEN (followers = 0 AND playlist_id !='') OR playlist_id = 'releaseRadar' OR playlist_id = 'dailyMix' THEN 'recommendation'
                        WHEN followers > 150000 THEN 'high'
                        WHEN followers > 10000 AND followers <= 150000 THEN 'mid'
                        WHEN followers > 0 AND followers <= 10000 THEN 'low'
                    END AS follower_category
                FROM spotify_detailed_playlist_data
            ),
            spotify_history_data AS (
                -- Extract historical playlist position and related data for each ISRC
                SELECT 
                    th.info, 
                    DATE_TRUNC('day', th.timestamp) AS streams_date, 
                    tr.isrc, 
                    ts.playlist_id,
                    CASE 
                        WHEN th.info LIKE 'Added to playlist at position%' THEN CAST(IFNULL(REGEXP_SUBSTR(th.info, '[0-9]+$'), 0) AS INTEGER)
                        WHEN th.info LIKE 'Position changed from%' THEN CAST(IFNULL(REGEXP_SUBSTR(th.info, '[0-9]+$'), 0) AS INTEGER)
                        ELSE NULL
                    END AS position
                FROM SPOTIFY_PLAYLIST_TRACK_HISTORY_SOURCE th
                JOIN SPOTIFY_PLAYLIST_TRACKS_SOURCE ts ON ts.ID = th.spotify_playlist_track_id
                JOIN prod_dwh.staging.uma_tracks_source tr ON tr.track_id = ts.track_id
                JOIN earliest_album_per_isrc ea ON ea.isrc = tr.isrc
            ),
            playlist_metrics AS (
                -- Calculate metrics for each ISRC per follower category per date using LEFT JOIN
                SELECT
                    psc.follower_category,
                    DATE_TRUNC('month', psc.streams_date) AS date,
                    psc.isrc,
                    AVG(shd.position) AS avg_position,
                    COUNT(DISTINCT psc.playlist_id) AS num_playlists,
                    SUM(psc.streams_playlist) AS total_streams
                FROM playlist_stream_categories psc
                LEFT JOIN spotify_history_data shd ON psc.isrc = shd.isrc
                    AND psc.playlist_id = shd.playlist_id
                    AND psc.streams_date = shd.streams_date
                GROUP BY psc.follower_category, DATE_TRUNC('month', psc.streams_date), psc.isrc
            )
            -- Final Selection
            SELECT 
                isrc,
                follower_category,
                date,
                total_streams,
                num_playlists,
                avg_position
            FROM playlist_metrics
            ORDER BY date, follower_category, isrc;
                """
        tracks_playlists_data = get_query(
            "snowflake", "eda_tracks_playlists_" + str(count), query
        )
        tracks_playlists_data.columns = [
            x.lower() for x in tracks_playlists_data.columns
        ]
        tracks_playlists_data["date"] = tracks_playlists_data["date"].apply(
            pd.to_datetime
        )
        tracks_playlists_data.set_index("date", inplace=True)
        return tracks_playlists_data

    def get_discovery_data(self, df):
        isrc, count = self.get_isrc_list(df)
        query = f"""
                WITH discovery_totals AS (
                    SELECT
                        ISRC,
                        DATE_TRUNC('day', date) AS discovery_date,
                        SUM(CASE WHEN discovery_key = TRUE THEN streams_total ELSE 0 END) AS total_discovery_streams
                    FROM PROD_DWH.DWH.ANALYTICS_DETAILED_SPOTIFY_DISCOVERY
                    WHERE discovery_key = TRUE
                    GROUP BY ISRC, DATE_TRUNC('day', date)
                ),
                filtered_isrcs AS (
                    SELECT DISTINCT
                        isrc
                    FROM PROD_DWH.STAGING.uma_tracks_source
                    WHERE isrc IN {(isrc)}
                ),
                length_data AS (
                    SELECT
                        ISRC,
                        DATE_TRUNC('day', date) AS length_date,
                        CASE 
                            WHEN LENGTH_KEY = '0-30' THEN 1 / 15.0
                            WHEN LENGTH_KEY = '31-60' THEN 1 / 45.5
                            WHEN LENGTH_KEY = '61-90' THEN 1 / 75.5
                            WHEN LENGTH_KEY = '91-120' THEN 1 / 105.5
                            WHEN LENGTH_KEY = '121-150' THEN 1 / 135.5
                            WHEN LENGTH_KEY = '151-180' THEN 1 / 165.5
                            WHEN LENGTH_KEY = '>180' THEN 1 / 200.0
                            ELSE 0
                        END AS inverse_length_value,
                        streams_total
                    FROM PROD_DWH.DWH.ANALYTICS_DETAILED_SPOTIFY_LENGTH
                ),
                length_aggregated AS (
                    SELECT
                        ISRC,
                        length_date,
                        SUM(inverse_length_value * streams_total) / NULLIF(SUM(streams_total), 0) AS avg_inverse_length
                    FROM length_data
                    GROUP BY ISRC, length_date
                ),
                age_data AS (
                    SELECT
                        ISRC,
                        DATE_TRUNC('day', date) AS age_date,
                        CASE 
                            WHEN AGE_KEY = '0-17' THEN 1 / 10.0
                            WHEN AGE_KEY = '18-24' THEN 1 / 21.0
                            WHEN AGE_KEY = '25-34' THEN 1 / 29.5
                            WHEN AGE_KEY = '35-44' THEN 1 / 39.5
                            WHEN AGE_KEY = '45-59' THEN 1 / 52.0
                            WHEN AGE_KEY = '60-150' THEN 1 / 80.0
                            ELSE 0
                        END AS inverse_age_value,
                        streams_total
                    FROM PROD_DWH.DWH.ANALYTICS_DETAILED_SPOTIFY_AGE
                ),
                age_aggregated AS (
                    SELECT
                        ISRC,
                        age_date,
                        SUM(inverse_age_value * streams_total) / NULLIF(SUM(streams_total), 0) AS avg_inverse_age
                    FROM age_data
                    GROUP BY ISRC, age_date
                ),
                monthly_data AS (
                    -- Combine discovery, length, and age data, and group by month
                    SELECT 
                        DATE_TRUNC('month', dt.discovery_date) AS month,
                        dt.isrc,
                        SUM(dt.total_discovery_streams) AS total_discovery_streams,
                        AVG(la.avg_inverse_length) AS avg_inverse_length,
                        AVG(aa.avg_inverse_age) AS avg_inverse_age,
                        MIN(dt.discovery_date) AS first_day_in_month
                    FROM discovery_totals dt
                    JOIN filtered_isrcs fi ON dt.isrc = fi.isrc
                    LEFT JOIN length_aggregated la ON la.isrc = dt.isrc AND la.length_date = dt.discovery_date
                    LEFT JOIN age_aggregated aa ON aa.isrc = dt.isrc AND aa.age_date = dt.discovery_date
                    GROUP BY DATE_TRUNC('month', dt.discovery_date), dt.isrc
                ),
                final_data AS (
                    -- Add the "first_month_incomplete" flag
                    SELECT 
                        month,
                        isrc,
                        total_discovery_streams,
                        avg_inverse_length,
                        avg_inverse_age,
                        CASE 
                            WHEN DAY(first_day_in_month) != 1 THEN 1
                            ELSE 0
                        END AS first_month_incomplete
                    FROM monthly_data
                )
                SELECT 
                    month as date,
                    isrc,
                    total_discovery_streams,
                    avg_inverse_length,
                    avg_inverse_age,
                    first_month_incomplete
                FROM final_data
                ORDER BY month, isrc;
"""
        tracks_discovery_data = get_query(
            "snowflake", "eda_tracks_discovery_" + str(count), query
        )
        tracks_discovery_data.columns = [
            x.lower() for x in tracks_discovery_data.columns
        ]
        tracks_discovery_data["date"] = tracks_discovery_data["date"].apply(
            pd.to_datetime
        )
        tracks_discovery_data.set_index("date", inplace=True)
        return tracks_discovery_data

    def get_artist_data(self, df):
        """
        Get artists related data based on given ISRC
        """
        artists, count = self.get_artist_list(df)
        query = f"""
                WITH tracks AS (
                    SELECT 
                        ut.title, 
                        ut.genre_id, 
                        ut.track_id, 
                        ut.label_id, 
                        ut.album_id, 
                        ut.performer, 
                        ut.explicit, 
                        ut.duration, 
                        ut.isrc,
                        ROW_NUMBER() OVER (PARTITION BY ut.isrc ORDER BY ut.isrc DESC) AS rn
                    FROM uma_tracks_source ut
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
                artists_tracks AS (
                    SELECT 
                        DATE_TRUNC('day', sa.date) AS date, 
                        a.artist AS performer, 
                        a.artist_id AS performer_id, 
                        ut.isrc, 
                        SUM(sa.streams_total) AS total_streams, 
                        COUNT(DISTINCT ut.track_id) AS track_count
                    FROM spotify_streams_aggregated sa
                    JOIN tracks ut ON ut.isrc = sa.isrc AND ut.rn = 1
                    JOIN artists a ON a.isrc = sa.isrc AND a.rn = 1
                    WHERE 
                        sa.isrc IS NOT NULL 
                        AND sa.isrc != '' 
                        AND a.artist_id IN {(artists)}  -- Artist filter, will be expanded for multiple artists in the list
                    GROUP BY 
                        DATE_TRUNC('day', sa.date), 
                        a.artist, 
                        a.artist_id, 
                        ut.isrc
                    ORDER BY date
                ),
                artists_with_seq AS (
                    SELECT 
                        at.*, 
                        COUNT_IF(seqnum = 1) AS tracks_cumcount 
                    FROM (
                        SELECT 
                            artists_tracks.*, 
                            ROW_NUMBER() OVER (PARTITION BY isrc ORDER BY date) AS seqnum
                        FROM artists_tracks
                    ) at
                    GROUP BY 
                        at.date, 
                        at.performer, 
                        at.performer_id, 
                        at.isrc, 
                        at.total_streams, 
                        at.track_count, 
                        seqnum
                    ORDER BY date, isrc
                ),
                preaggregated_artists AS (
                    SELECT 
                        date, 
                        performer, 
                        performer_id, 
                        total_streams, 
                        track_count,
                        AVG(total_streams) OVER (PARTITION BY performer_id ORDER BY date) AS streams_avg, 
                        SUM(tracks_cumcount) OVER (PARTITION BY performer_id ORDER BY date) AS tracks_cumsum
                    FROM artists_with_seq
                )
                SELECT 
                    date, 
                    performer, 
                    performer_id AS artist_id, 
                    SUM(total_streams) AS artist_total_streams, 
                    SUM(track_count) AS artist_track_count, 
                    MIN(streams_avg) AS artist_streams_avg_to_date, 
                FROM preaggregated_artists
                GROUP BY date, performer, performer_id
                ORDER BY date, performer;
            """
        artist_data = get_query("snowflake", "artists_" + str(count), query)
        artist_data.columns = [x.lower() for x in artist_data.columns]
        artist_data["date"] = artist_data["date"].apply(pd.to_datetime)
        artist_data = artist_data.drop_duplicates(
            ["date", "artist_id"]
        )  # there are duplicates sometimes with the artist
        artist_data.set_index("date", inplace=True)
        return artist_data

    def get_genre_data(self, df):
        """
        Get genre related data based on given ISRC
        """
        genres, count = self.get_genre_list(df)
        query = f"""
            WITH tracks AS (
                SELECT 
                    ut.title, 
                    ut.genre_id, 
                    ut.track_id, 
                    ut.label_id, 
                    ut.album_id, 
                    ut.performer, 
                    ut.explicit, 
                    ut.duration, 
                    ut.isrc,
                    ROW_NUMBER() OVER (PARTITION BY ut.isrc ORDER BY ut.isrc DESC) AS rn
                FROM uma_tracks_source ut
            ), 
            genre_tracks AS (
                SELECT 
                    DATE_TRUNC('day', sa.date) AS date, 
                    gs.genre_id, 
                    gs.genre_title AS genre, 
                    ut.isrc, 
                    SUM(sa.streams_total) AS total_streams
                FROM spotify_streams_aggregated sa
                JOIN tracks ut ON ut.isrc = sa.isrc AND ut.rn = 1
                JOIN ingrooves_genre_source gs ON ut.genre_id = gs.genre_id
                WHERE 
                    sa.isrc IS NOT NULL 
                    AND sa.isrc != '' 
                    AND gs.genre_id IN {(genres)}  -- Use your list of genres here
                GROUP BY 
                    DATE_TRUNC('day', sa.date), 
                    gs.genre_id, 
                    gs.genre_title, 
                    ut.isrc
                ORDER BY date
            ),
            genre_with_seq AS (
                SELECT 
                    at.*, 
                    COUNT_IF(seqnum = 1) AS track_cumcount
                FROM (
                    SELECT 
                        genre_tracks.*, 
                        ROW_NUMBER() OVER (PARTITION BY isrc ORDER BY date) AS seqnum
                    FROM genre_tracks
                ) at
                GROUP BY 
                    at.date, 
                    at.genre_id, 
                    at.genre, 
                    at.isrc, 
                    at.total_streams, 
                    seqnum
                ORDER BY date, isrc
            ),
            preaggregated_genres AS (
                SELECT 
                    date, 
                    genre, 
                    genre_id, 
                    total_streams,
                    AVG(total_streams) OVER (PARTITION BY genre_id ORDER BY date) AS streams_avg,
                    SUM(track_cumcount) OVER (PARTITION BY genre_id ORDER BY date) AS track_cumsum
                FROM genre_with_seq
            )
            SELECT 
                date, 
                genre, 
                genre_id, 
                SUM(total_streams) AS genre_total_streams, 
                MIN(streams_avg) AS genre_streams_avg_to_date, 
                MIN(track_cumsum) AS genre_track_count
            FROM preaggregated_genres
            GROUP BY date, genre, genre_id
            ORDER BY date, genre;
        """
        genre_data = get_query("snowflake", "genre_" + str(count), query)
        genre_data.columns = [x.lower() for x in genre_data.columns]
        genre_data["date"] = genre_data["date"].apply(pd.to_datetime)
        genre_data.set_index("date", inplace=True)
        return genre_data

    def get_label_data(self, df):
        labels, count = self.get_label_list(df)
        query = f"""
            WITH tracks AS (
                SELECT 
                    ut.title, 
                    ut.genre_id, 
                    ut.track_id, 
                    ut.label_id, 
                    ut.album_id, 
                    ut.performer, 
                    ut.explicit, 
                    ut.duration, 
                    ut.isrc,
                    ROW_NUMBER() OVER (PARTITION BY ut.isrc ORDER BY ut.isrc DESC) AS rn
                FROM uma_tracks_source ut
            ), 
            label_tracks AS (
                SELECT 
                    DATE_TRUNC('day', sa.date) AS date, 
                    lbs.label_id,
                    lbs.label_name, 
                    ut.isrc, 
                    SUM(sa.streams_total) AS total_streams
                FROM spotify_streams_aggregated sa
                JOIN tracks ut ON ut.isrc = sa.isrc AND ut.rn = 1
                JOIN UMA_LABELS_SOURCE lbs ON ut.label_id = lbs.label_id
                WHERE 
                    sa.isrc IS NOT NULL 
                    AND sa.isrc != ''
                    and lbs.label_id IN {(labels)}
                GROUP BY 
                    DATE_TRUNC('day', sa.date), 
                    lbs.label_id,
                    lbs.label_name, 
                    ut.isrc
                ORDER BY date
            ),
            label_with_seq AS (
                SELECT 
                    at.*, 
                    COUNT_IF(seqnum = 1) AS track_cumcount
                FROM (
                    SELECT 
                        label_tracks.*, 
                        ROW_NUMBER() OVER (PARTITION BY isrc ORDER BY date) AS seqnum
                    FROM label_tracks
                ) at
                GROUP BY 
                    at.date, 
                    at.label_id, 
                    at.label_name, 
                    at.isrc, 
                    at.total_streams, 
                    seqnum
                ORDER BY date, isrc
            ),
            preaggregated_labels AS (
                SELECT 
                    date, 
                    label_name, 
                    label_id, 
                    total_streams,
                    AVG(total_streams) OVER (PARTITION BY label_id ORDER BY date) AS streams_avg,
                    SUM(track_cumcount) OVER (PARTITION BY label_id ORDER BY date) AS track_cumsum
                FROM label_with_seq
            )
            SELECT 
                date, 
                label_name, 
                label_id, 
                SUM(total_streams) AS label_total_streams, 
                MIN(streams_avg) AS label_streams_avg_to_date, 
                MIN(track_cumsum) AS label_track_count
            FROM preaggregated_labels
            GROUP BY date, label_name, label_id
            ORDER BY date, label_name;
"""
        label_data = get_query("snowflake", "label_" + str(count), query)
        label_data.columns = [x.lower() for x in label_data.columns]
        label_data["date"] = label_data["date"].apply(pd.to_datetime)
        label_data.set_index("date", inplace=True)
        return label_data

    def get_playlists_data(self, df):
        playlists, count = self.get_playlist_list(df)
        query = f"""
            WITH unique_tracks AS (
                -- Extract unique track data
                SELECT 
                    tr.track_id,
                    tr.isrc,
                    api.ACOUSTICNESS,
                    api.DANCEABILITY,
                    api.ENERGY,
                    api.INSTRUMENTALNESS,
                    api.LIVENESS,
                    api.SPEECHINESS,
                    api.VALENCE,
                    api.LOUDNESS,
                    api.TEMPO,
                    api.KEY,
                    api.MODE,
                    st.streams_total AS track_streams,
                    tr.genre_id,
                    al.release_date,
                    al.explicit_lyrics
                FROM prod_dwh.staging.uma_tracks_source tr
                LEFT JOIN PROD_DWH.STAGING.SPOTIFY_TRACK_API_ANALYTICS api ON api.track_id = tr.track_id
                LEFT JOIN spotify_streams_aggregated st ON st.isrc = tr.isrc
                LEFT JOIN prod_dwh.staging.uma_albums_source al ON al.album_id = tr.album_id
            ),
            playlist_tracks AS (
                -- Extract playlist and track relationships uniquely
                SELECT DISTINCT
                    pl.playlist_id,
                    tr.track_id
                FROM SPOTIFY_PLAYLIST_TRACKS_SOURCE pl
                JOIN prod_dwh.staging.uma_tracks_source tr ON tr.track_id = pl.track_id
            ),
            playlist_track_features AS (
                -- Join unique track data with playlist information
                SELECT 
                    pt.playlist_id,
                    ut.track_id,
                    ut.isrc,
                    ut.acousticness,
                    ut.danceability,
                    ut.energy,
                    ut.instrumentalness,
                    ut.liveness,
                    ut.speechiness,
                    ut.valence,
                    ut.loudness,
                    ut.tempo,
                    ut.key,
                    ut.mode,
                    ut.track_streams,
                    ut.genre_id,
                    ut.release_date,
                    ut.explicit_lyrics
                FROM playlist_tracks pt
                JOIN unique_tracks ut ON pt.track_id = ut.track_id
            ),
            playlist_features_aggregated AS (
                -- Aggregate features for each playlist
                SELECT 
                    playlist_id,
                    COUNT(DISTINCT track_id) AS num_tracks,
                    AVG(acousticness) AS avg_acousticness,
                    AVG(danceability) AS avg_danceability,
                    AVG(energy) AS avg_energy,
                    AVG(instrumentalness) AS avg_instrumentalness,
                    AVG(liveness) AS avg_liveness,
                    AVG(speechiness) AS avg_speechiness,
                    AVG(valence) AS avg_valence,
                    AVG(loudness) AS avg_loudness,
                    AVG(tempo) AS avg_tempo,
                    AVG(key) AS avg_key,
                    AVG(mode) AS avg_mode,
                    -- Total streams, average streams, median streams, and top track contribution
                    SUM(track_streams) AS total_playlist_streams,
                    AVG(track_streams) AS avg_streams_per_track,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY track_streams) AS median_streams_per_track,
                    MAX(track_streams) * 1.0 / NULLIF(SUM(track_streams), 0) AS top_track_contribution_ratio,
                    -- Genre diversity features
                    COUNT(DISTINCT genre_id) AS num_unique_genres,
                    -- Track age features
                    AVG(EXTRACT(YEAR FROM release_date)) AS avg_release_year,
                    EXTRACT(YEAR FROM MAX(release_date)) - EXTRACT(YEAR FROM MIN(release_date)) AS age_spread,
                    -- Explicit content ratio
                    SUM(CASE WHEN explicit_lyrics = TRUE THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS explicit_content_ratio
                FROM playlist_track_features
                GROUP BY playlist_id
            ),
            playlist_metadata AS (
                -- Extract metadata for each playlist, including follower count and playlist name
                SELECT 
                    pl_meta.PLAYLIST_ID AS playlist_id,
                    MAX(pl_meta.followers) AS max_followers,
                    pl_meta.name
                FROM PROD_DWH.STAGING.SPOTIFY_PLAYLISTS_SOURCE pl_meta
                GROUP BY pl_meta.PLAYLIST_ID, pl_meta.name
            ),
            playlist_data_combined AS (
                -- Combine playlist-level audio features, metadata, and additional metrics
                SELECT 
                    pf.playlist_id,
                    pf.num_tracks,
                    pf.avg_acousticness,
                    pf.avg_danceability,
                    pf.avg_energy,
                    pf.avg_instrumentalness,
                    pf.avg_liveness,
                    pf.avg_speechiness,
                    pf.avg_valence,
                    pf.avg_loudness,
                    pf.avg_tempo,
                    pf.avg_key,
                    pf.avg_mode,
                    pf.total_playlist_streams,
                    pf.avg_streams_per_track,
                    pf.median_streams_per_track,
                    pf.top_track_contribution_ratio,
                    pf.num_unique_genres,
                    pf.avg_release_year,
                    pf.age_spread,
                    pf.explicit_content_ratio,
                    pm.max_followers,
                    pm.name
                FROM playlist_features_aggregated pf
                LEFT JOIN playlist_metadata pm ON pf.playlist_id = pm.playlist_id
            )
            -- Final select for data gathering
            SELECT *
            FROM playlist_data_combined
            where playlist_id IN {(playlists)};"""
        playlist_data = get_query("snowflake", "playlist_" + str(count), query)
        playlist_data.columns = [x.lower() for x in playlist_data.columns]
        return playlist_data

    def get_tracks_features(self, df):
        isrc, count = self.get_isrc_list(df)
        query = f"""SELECT t.ISRC, ACOUSTICNESS, DANCEABILITY, DURATION_MS, ENERGY, INSTRUMENTALNESS, 
                LIVENESS, SPEECHINESS, VALENCE, END_OF_FADE_IN, START_OF_FADE_OUT, LOUDNESS, TEMPO, TEMPO_CONFIDENCE, 
                TIME_SIGNATURE, TIME_SIGNATURE_CONFIDENCE, KEY, KEY_CONFIDENCE, MODE, MODE_CONFIDENCE
                FROM spotify_track_api_analytics aa
                join uma_tracks_source t on t.track_id = aa.track_id 
                where t.ISRC IN {(isrc)}"""
        tracks_features = get_query("snowflake", "track_features_" + str(count), query)
        tracks_features.columns = [x.lower() for x in tracks_features.columns]
        return tracks_features

    def get_music_data(self) -> EdaMusicDataModel:
        if self._music_data:
            return self._music_data
        tracks_df = self.get_tracks_by_popularity()
        self._music_data = EdaMusicDataModel(tracks=tracks_df)
        playlists_df = self.get_tracks_in_playlists_data(tracks_df)
        self._music_data.playlists = playlists_df
        discovery_df = self.get_discovery_data(tracks_df)
        self._music_data.discovery = discovery_df
        # artist_df = self.get_artist_data(tracks_df)
        # self._music_data.artists = artist_df
        # genre_df = self.get_genre_data(tracks_df)
        # self._music_data.genres = genre_df
        # labels_df = self.get_label_data(tracks_df)
        # self._music_data.labels = labels_df
        # playlists_df = self.get_playlists_data(tracks_df)
        # self._music_data.playlists = playlists_df
        return self._music_data

    def get_merged_data2(self, music_data=None, cutoff_date="2024-08-31"):
        if music_data is not None:
            self._music_data = music_data
        if not self._music_data:
            self.get_music_data()
        # Filter playlists by the cutoff date
        playlists = self._music_data.playlists[
            self._music_data.playlists.index <= pd.to_datetime(cutoff_date)
        ].copy()
        tracks = self._music_data.tracks.copy()
        discovery = self._music_data.discovery[
            self._music_data.discovery.index <= pd.to_datetime(cutoff_date)
        ].copy()

        # Reset index for playlists
        playlists.reset_index(inplace=True)
        discovery.reset_index(inplace=True)
        # Filtering the rows with 'follower_category' equal to 'high' to calculate the averages
        high_category_data = (
            playlists[playlists["follower_category"] == "high"]
            .groupby(["isrc", "date"])
            .agg({"num_playlists": lambda x: x.mean(skipna=True)})
            .reset_index()
        )
        # Pivoting the 'follower_category' to columns with 'total_streams' as the values
        pivoted_data = playlists.pivot_table(
            index=["isrc", "date"],
            columns="follower_category",
            values="total_streams",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        pivoted_data = pivoted_data.merge(
            high_category_data, on=["isrc", "date"], how="left"
        )
        # Calculating total streams from all follower categories
        pivoted_data["total_streams"] = pivoted_data.sum(axis=1, numeric_only=True)

        # Calculating proportions for each follower category
        for column in pivoted_data.columns:
            if column not in [
                "isrc",
                "date",
                "total_streams",
                "num_playlists",
                "avg_position",
            ]:
                pivoted_data[f"proportion_{column}"] = (
                    pivoted_data[column] / pivoted_data["total_streams"]
                )

        # Merge playlists and tracks dataframes
        merged_df = pivoted_data.merge(
            tracks,
            left_on=["isrc"],
            right_on=["isrc"],
            how="left",
        )
        merged_df = merged_df.merge(
            discovery,
            left_on=["date", "isrc"],
            right_on=["date", "isrc"],
            how="left",
        )
        # Rename columns example
        merged_df.rename(
            columns={
                "total_streams_x": "streams",
                "total_streams_y": "overall_streams",
            },
            inplace=True,
        )
        # Fill missing values with 0
        df = merged_df.fillna(0)

        # Sort by 'date' and 'isrc' columns
        df = df.sort_values(by=["date", "isrc"]).reset_index(drop=True)

        # Create unique day dataframe and calculate 'age'
        df_unique_day = df[["isrc", "date"]].drop_duplicates()
        df_unique_day["age"] = df_unique_day.groupby("isrc").cumcount() + 1

        # Merge 'age' back to the main dataframe
        df = df.merge(df_unique_day, on=["isrc", "date"], how="left")

        # Drop the original 'date' column and reorder columns to have 'age' first
        df = df.drop(columns=["date"])
        cols = df.columns.tolist()
        reordered_cols = [cols[-1]] + cols[:-1]
        df = df[reordered_cols]

        # Calculate the global maximum age across all ISRCs
        max_age = df["age"].max()
        max_age_per_isrc = (
            df.groupby("isrc")["age"]
            .max()
            .reset_index()
            .rename(columns={"age": "max_age_per_isrc"})
        )
        df = df.merge(max_age_per_isrc, on="isrc", how="left")

        # Create 'censored' column
        df["censored"] = (
            (df["age"] == df["max_age_per_isrc"]) & (df["max_age_per_isrc"] < max_age)
        ).astype(int)

        # Drop the 'max_age_per_isrc' column
        df.drop(columns=["max_age_per_isrc"], inplace=True)
        df["proportion_discovery"] = df["total_discovery_streams"] / df["streams"]
        popularity_one_hot = pd.get_dummies(df["popularity_category"], prefix="track")
        df.drop(columns=["popularity_category"], inplace=True)
        # print(popularity_one_hot.columns)
        df = df.join(popularity_one_hot)
        return df

    def get_merged_data(self, music_data=None):
        if music_data is not None:
            self._music_data = music_data
        if not self._music_data:
            self.get_music_data()
        tracks_df_expanded = self._music_data.tracks.copy()
        artists_df = self._music_data.artists.copy()
        label_df = self._music_data.labels.copy()
        genre_df = (
            self._music_data.genres.copy()
        )  # Assume this DataFrame contains genre data
        playlist_df = self._music_data.playlists.copy()

        tracks_df_expanded["original_artist_id"] = tracks_df_expanded["artist_id"]

        tracks_df_expanded["artist_id"] = (
            tracks_df_expanded["artist_id"].astype(str).str.split(",")
        )
        # Step 2: Expand the artist IDs into multiple rows
        tracks_df_expanded = tracks_df_expanded.explode("artist_id")
        # Convert artist_id to numeric to match the type in data_artists
        tracks_df_expanded["artist_id"] = tracks_df_expanded["artist_id"].astype(int)
        tracks_df_expanded["label_id"] = tracks_df_expanded["label_id"].astype(int)
        tracks_df_expanded["explicit"] = tracks_df_expanded["explicit"].astype(int)
        artists_df["artist_id"] = artists_df["artist_id"].astype(int)
        # Step 2: Merge expanded tracks data with artists data on index (date) and artist_id
        merged_df = tracks_df_expanded.merge(
            artists_df,
            left_on=["date", "artist_id"],
            right_on=["date", "artist_id"],
            how="left",
        )
        merged_df = merged_df.merge(
            genre_df,
            left_on=["date", "genre_id"],
            right_on=["date", "genre_id"],
            how="left",
        )
        merged_df = merged_df.merge(
            label_df,
            left_on=["date", "label_id"],
            right_on=["date", "label_id"],
            how="left",
        )

        # Step to update merging process for additional insights
        merged_df.reset_index(inplace=True)
        merged_df = merged_df.merge(
            playlist_df,  # Add playlist data for calculating distinctiveness
            left_on="playlist_id",
            right_on="playlist_id",
            how="left",
        )
        merged_df.set_index("date", inplace=True)
        merged_aggregated_df = (
            merged_df.groupby(["isrc", "date", "playlist_id"])
            .agg(
                {
                    "streams_playlist": "first",
                    "followers": "first",
                    "spotify_streams": "first",
                    "position": "first",
                    "event": "first",
                    "performer": lambda x: ", ".join(
                        sorted(set(str(name) for name in x if pd.notnull(name)))
                    ),
                    "artist_total_streams": "sum",
                    "artist_track_count": "sum",
                    "artist_streams_avg_to_date": "sum",
                    "genre": "first",
                    "genre_total_streams": "sum",
                    "genre_track_count": "sum",
                    "genre_streams_avg_to_date": "sum",
                    "label_name": "first",
                    "label_total_streams": "sum",
                    "label_track_count": "sum",
                    "label_streams_avg_to_date": "sum",
                    "release_date": "first",
                    "featured_artist": "first",
                    "explicit": "first",
                    "num_tracks": "first",
                    "avg_acousticness": "first",
                    "avg_danceability": "first",
                    "avg_energy": "first",
                    "avg_instrumentalness": "first",
                    "avg_liveness": "first",
                    "avg_speechiness": "first",
                    "avg_valence": "first",
                    "avg_loudness": "first",
                    "avg_tempo": "first",
                    "avg_key": "first",
                    "avg_mode": "first",
                    "total_playlist_streams": "first",
                    "avg_streams_per_track": "first",
                    "median_streams_per_track": "first",
                    "top_track_contribution_ratio": "first",
                    "num_unique_genres": "first",
                    "avg_release_year": "first",
                    "age_spread": "first",
                    "explicit_content_ratio": "first",
                    "max_followers": "first",
                }
            )
            .reset_index()
        )
        merged_aggregated_df.set_index("date", inplace=True)
        merged_aggregated_df["release_date"] = pd.to_datetime(
            merged_aggregated_df["release_date"]
        )
        self._merged_data = merged_aggregated_df
        return self._merged_data

    def get_featured_dataset(self):
        dataset = self._merged_data.copy()
        dataset["release_date"] = pd.to_datetime(dataset["release_date"])

        # Calculate overall values for distinctiveness metrics
        overall_avg_instrumentalness = float(dataset["avg_instrumentalness"].mean())
        overall_avg_valence = float(dataset["avg_valence"].mean())
        overall_explicit_ratio = float(dataset["explicit"].mean())
        avg_release_year_all = float(dataset["avg_release_year"].mean())
        # Initialize lists to store calculated features for each group
        results = []

        # Iterate over each group (by `isrc` and `playlist_id`)
        for (isrc, playlist_id), group in dataset.groupby(["isrc", "playlist_id"]):
            # Sort by date within the group to maintain time order (date is index)
            group = group.sort_index()

            # Time-Based Features
            first_date = group.index[0]
            last_date = group.index[-1]
            days_on_playlist = (last_date - first_date).days
            days_since_release = (first_date - group["release_date"]).dt.days

            # Calculate lag features for streams_playlist
            group["streams_lag_1"] = group["streams_playlist"].shift(1).fillna(0)
            group["streams_lag_7"] = group["streams_playlist"].shift(7).fillna(0)
            group["position_lag_1"] = group["position"].shift(1).fillna(0)
            group["position_lag_7"] = group["position"].shift(7).fillna(0)
            # Calculate cumulative streams and rolling statistics

            group["cumulative_playlist_streams"] = group["streams_playlist"].cumsum()
            group["playlist_growth_rate"] = (
                group["streams_playlist"].pct_change().fillna(0)
            )
            group["rolling_avg_streams"] = (
                group["streams_playlist"].rolling(window=7, min_periods=1).mean()
            )
            group["stream_acceleration"] = (
                group["playlist_growth_rate"].diff().fillna(0)
            )

            # Peak and relative statistics
            peak_position = group.loc[group["position"] > 0, "position"].min()
            peak_position = int(peak_position) if pd.notna(peak_position) else 0
            last_position = group["position"].max()
            relative_popularity = group.apply(
                lambda row: (
                    row["streams_playlist"] / row["genre_total_streams"]
                    if row["genre_total_streams"] != 0
                    else 0
                ),
                axis=1,
            )

            # Statistical Features
            mean_streams = group["streams_playlist"].mean()
            variance_streams = np.nan_to_num(group["streams_playlist"].var(), nan=0)
            skewness_streams = np.nan_to_num(group["streams_playlist"].skew(), nan=0)
            kurtosis_streams = (
                group["streams_playlist"].kurtosis()
                if len(group["streams_playlist"]) > 4
                else 0
            )  # Kurtosis should have at least 4 datapoints

            # Other features
            engagement_ratio = group["streams_playlist"] / group["followers"]
            artist_genre_performance = (
                group["artist_streams_avg_to_date"] / group["genre_streams_avg_to_date"]
            )
            artist_label_performance = group["artist_streams_avg_to_date"].astype(
                float
            ) / group["label_streams_avg_to_date"].astype(float)
            track_artist_ratio = group.apply(
                lambda row: (
                    row["streams_playlist"] / row["artist_streams_avg_to_date"]
                    if row["artist_streams_avg_to_date"] != 0
                    else 0
                ),
                axis=1,
            )
            position_changes = (
                group["position"].diff().ne(0).sum()
            )  # Count of position changes
            streams_followers_interaction = (
                group["streams_playlist"] * group["followers"]
            )
            explicit_streams = group["explicit"] * group["streams_playlist"]
            artist_count = (
                group["performer"].astype(str).apply(lambda x: len(x.split(",")))
            )

            age_distinctiveness = abs(
                float(group["avg_release_year"].iloc[0]) - float(avg_release_year_all)
            )
            instrumentalness_distinctiveness = float(
                group["avg_instrumentalness"].iloc[0]
            ) - float(overall_avg_instrumentalness)
            valence_distinctiveness = group["avg_valence"].iloc[0] - overall_avg_valence
            explicit_distinctiveness = group["explicit"].mean() - overall_explicit_ratio
            streams_popularity_ratio = (
                group["streams_playlist"].iloc[0] / group["followers"].iloc[0]
            )

            top_track_contribution_ratio = (
                group["streams_playlist"].max() / group["streams_playlist"].sum()
            )

            # Append computed statistics for the group
            results.append(
                {
                    "isrc": isrc,
                    "playlist_id": playlist_id,
                    "days_on_playlist": days_on_playlist,
                    "days_since_release": days_since_release.iloc[
                        -1
                    ],  # Final days since release in group
                    "streams_lag_1": group["streams_lag_1"].iloc[-1],
                    "streams_lag_7": group["streams_lag_7"].iloc[-1],
                    "position_lag_1": group["position_lag_1"].iloc[-1],
                    "position_lag_7": group["position_lag_7"].iloc[-1],
                    "playlist_cumulative_streams": group[
                        "cumulative_playlist_streams"
                    ].iloc[-1],
                    # Cumulative streams up to the last date
                    "playlist_growth_rate": group["playlist_growth_rate"].mean(),
                    "playlist_streams_rolling_avg": group["rolling_avg_streams"].mean(),
                    "stream_acceleration": group["stream_acceleration"].mean(),
                    "peak_position": peak_position,
                    "last_position": last_position,
                    "relative_popularity_to_genre": relative_popularity.mean(),
                    "mean_streams": mean_streams,
                    "variance_streams": variance_streams,
                    "skewness_streams": skewness_streams,
                    "kurtosis_streams": kurtosis_streams,
                    "engagement_ratio": engagement_ratio.mean(),
                    "artist_genre_performance": artist_genre_performance.mean(),
                    "artist_label_performance": artist_label_performance.mean(),
                    "track_artist_ratio": track_artist_ratio.mean(),
                    "position_changes": position_changes,
                    "streams_followers_interaction": streams_followers_interaction.sum(),
                    "explicit": group["explicit"].iloc[0],
                    "featured_artist": group["featured_artist"].iloc[0],
                    "artist_count": artist_count.iloc[0],
                    # Playlist distinctiveness features
                    "age_distinctiveness": age_distinctiveness,
                    "instrumentalness_distinctiveness": instrumentalness_distinctiveness,
                    "valence_distinctiveness": valence_distinctiveness,
                    "explicit_distinctiveness": explicit_distinctiveness,
                    "streams_popularity_ratio": streams_popularity_ratio,
                    "top_track_contribution_ratio": top_track_contribution_ratio,
                }
            )

        # Convert results into a DataFrame
        features_df = pd.DataFrame(results)
        return features_df


if __name__ == "__main__":
    data = MusicService().get_music_data()
    print(data.tracks.head())
    print(data.artists.head())
    print(data.genres.head())
