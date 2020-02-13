# IMPORTANT...pip install pytube3
from pytube import Playlist
more = True
while more:
    pl = input("URL of playlist: ")
    playlist = Playlist(pl)
    count = 0
    for video in playlist.videos:
        video.streams.get_highest_resolution().download()
    c = input("Continue adding playlists? (y/n): ").lower()
    if c == 'n':
        more = False

