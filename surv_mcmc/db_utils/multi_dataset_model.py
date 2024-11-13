import numpy as np

from db_utils.db_utils import get_query
import pandas as pd

from dataclasses import dataclass


@dataclass
class MusicDataModel:
    """Class to gather different music data for analysis
    Пример:
    data = MusicService().get_music_data(track_amount=50, track_offset=10, min_playlist_followers=5000000)
    """

    tracks: pd.DataFrame = None
    artists: pd.DataFrame = None
    genres: pd.DataFrame = None
    labels: pd.DataFrame = None
    playlists: pd.DataFrame = None


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

    def get_tracks_in_playlists_data(self):
        """
        Get best track data from database with given offset from the top
        """
        query = """
                    WITH spotify_streams_data AS (
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
                    WHERE ts.playlist_id IN ('6UeSakyzhiEt4NB3UAd6NQ','37i9dQZF1DWXT8uSSn6PRy','37i9dQZEVXbKCF6dqVpDkS','6fi0ExpvtWx0vQUelJkwmV','37i9dQZEVXbJPcfkRz0wJ0',
    '37i9dQZF1DWYs83FtTMQFw','37i9dQZEVXbJNSeeHswcKB','37i9dQZEVXbLRQDuF5jeBp','37i9dQZF1DX1hVRardJ30X','37i9dQZEVXbMDoHDwVN2tF','4OIVU71yO7SzyGrh0ils2i','37i9dQZEVXbLoATJ81JYXz',
    '37i9dQZEVXbJvfa0Yxg7E7','37i9dQZF1E4DTZUur7HqeC','37i9dQZEVXbIPWwFssbupI','37i9dQZF1DX9oh43oAzkyx','37i9dQZF1DX4y8h9WqDPAE','37i9dQZF1DXcBWIGoYBM5M',' 5WNUX6jisX0NRlVDTm6RDd',
    '37i9dQZF1DXcA6dRp8rwj6','37i9dQZF1DWZdsS73T1ogG','37i9dQZF1DX60OAKjsWlA2','37i9dQZF1DWU0zylzZL5LY','37i9dQZF1E4y5lWP0HJz8d','1NjQR0evUqgnVWOQVn8cci','37i9dQZEVXbNFJfN1Vw8d9',
    '37i9dQZF1DX45xYefy6tIi','37i9dQZEVXbObFQZ3JLcXt','37i9dQZF1DWTyiBJ6yEqeu','37i9dQZF1DWT2SPAYawYcO','37i9dQZEVXbMxcczTSoGwZ','5LwGTy1gOEG8SnvZb2s7pY','37i9dQZEVXbN6itCcaL3Tt',
    '37i9dQZEVXbL3J0k32lWnN') 
                      AND ea.earliest_release_date < '2024-05-01' 
                      AND ea.earliest_release_date > '2021-08-01'
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
                    shd.isrc, 
                    shd.playlist_id, 
                    shd.info, 
                    DATE_TRUNC(day, shd.timestamp) AS date, 
                    sp.streams_playlist, 
                    sp.followers, 
                    st.spotify_streams,
                    pdt.position,
                    pdt.event,
                    a.artist_id,
                    shd.genre_id,
                    shd.release_date,
                    pdt.featured_artist,
                    ads.explicit,
                    shd.label_id
                FROM spotify_history_data shd
                JOIN filtered_isrcs fi ON shd.isrc = fi.isrc  -- Join to keep only ISRCs with add_count < 3
                JOIN spotify_detailed_playlist_data sp ON sp.isrc = shd.isrc AND sp.streams_date = DATE_TRUNC(day, shd.timestamp) AND sp.playlist_id = shd.playlist_id
                JOIN explicit_info ads ON ads.isrc = fi.isrc AND ads.rn = 1
                JOIN artists a ON a.isrc = shd.isrc AND a.rn = 1
                LEFT JOIN spotify_streams_data st ON st.isrc = shd.isrc AND st.streams_date = DATE_TRUNC(day, shd.timestamp)
                LEFT JOIN playlist_data_transformed pdt ON pdt.playlist_id = shd.playlist_id AND DATE_TRUNC(day, pdt.timestamp) = DATE_TRUNC(day, shd.timestamp) AND pdt.isrc = shd.isrc
                ORDER BY shd.isrc, shd.timestamp;
                """
        track_data = get_query("snowflake", "tracks_rap_cavier", query)
        track_data.columns = [x.lower() for x in track_data.columns]
        track_data["date"] = track_data["date"].apply(pd.to_datetime)
        track_data.set_index("date", inplace=True)
        return track_data

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

    def get_music_data(self) -> MusicDataModel:
        if self._music_data:
            return self._music_data
        tracks_df = self.get_tracks_in_playlists_data()
        self._music_data = MusicDataModel(tracks=tracks_df)
        artist_df = self.get_artist_data(tracks_df)
        self._music_data.artists = artist_df
        genre_df = self.get_genre_data(tracks_df)
        self._music_data.genres = genre_df
        labels_df = self.get_label_data(tracks_df)
        self._music_data.labels = labels_df
        playlists_df = self.get_playlists_data(tracks_df)
        self._music_data.playlists = playlists_df
        return self._music_data

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
