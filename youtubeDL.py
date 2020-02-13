from pytube import YouTube
# urls = ["https://youtu.be/_lgVcodSjHM?list=PLCiOXwirraUDRk5TlB2ulS3V2-0tB3vcS","https://youtu.be/k3TNqA8tI8Y?list=PLCiOXwirraUDRk5TlB2ulS3V2-0tB3vcS", "https://youtu.be/QIff2wXoRSw?list=PLCiOXwirraUDRk5TlB2ulS3V2-0tB3vcS", "https://youtu.be/F3w0DOqhik4?list=PLCiOXwirraUDRk5TlB2ulS3V2-0tB3vcS", "https://youtu.be/YejAkhvh6N0?list=PLCiOXwirraUDRk5TlB2ulS3V2-0tB3vcS"]
urls = [
    "https://youtu.be/VqtNhHl8JwM?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/_qAqqqYcvOQ?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/JdwRQ8r-SGE?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/_KoEz9K6t9I?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/3u1df3FSdno?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/tt9DYh6RFrY?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/3LkjZ4DEmqk?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/ytNzrUJnfZI?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/zt_JvGbQUbE?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37",
    "https://youtu.be/0ohspbFwoks?list=PLCiOXwirraUAvkTPDWeeSqAKty3LAG37"
]
for url in urls:
    YouTube(url).streams.first().download()
