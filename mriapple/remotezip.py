"""Read single members of a large remote ZIP archive through HTTP range requests.

The TotalSegmentator dataset ships as one 23.6 GB archive. Only the chosen scans are
needed, so the central directory and those members are fetched by byte range.
"""
import http.client
import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

USER_AGENT = 'mriapple-fetch/1.0'


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP byte ranges, with a small read-ahead block."""
    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos = url, block, 0
        self.size = int(self._open('bytes=0-0').headers['Content-Range'].rsplit('/', 1)[1])
        self.cache_start, self.cache = 0, b''

    def _open(self, byte_range):
        request = urllib.request.Request(self.url, headers={'Range': byte_range, 'User-Agent': USER_AGENT})
        response = urllib.request.urlopen(request, timeout=120)
        if response.status != 206:
            raise OSError(f'Server ignored the byte range for {self.url}')
        return response

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b''
        end = self.pos + n
        if not (self.cache_start <= self.pos and end <= self.cache_start + len(self.cache)):
            fetch = max(n, self.block)
            last = min(self.size, self.pos + fetch) - 1
            self.cache, self.cache_start = fetch_range(self.url, self.pos, last), self.pos
        offset = self.pos - self.cache_start
        data = self.cache[offset:offset + n]
        self.pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def fetch_range(url, first, last, attempts=6):
    """Bytes first..last inclusive, retrying dropped connections."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}',
                                                           'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise OSError(f'Server ignored the byte range for {url}')
                data = response.read()
            if len(data) == last - first + 1:
                return data
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts - 1:
                raise
            wait = int(error.headers.get('Retry-After') or 60)
            print(f'Server asked to slow down; waiting {wait} s', flush=True)
            time.sleep(wait)
            continue
        except (http.client.IncompleteRead, ConnectionError, TimeoutError, OSError) as error:
            if attempt == attempts - 1:
                raise
            print(f'Retrying range {first}-{last}: {error}', flush=True)
        time.sleep(2**attempt)
    raise OSError(f'Incomplete range {first}-{last} from {url}')


def member_prefix(url, info, n):
    """First n uncompressed bytes of one member, using offsets from an already-read
    directory, without re-reading the archive's central directory."""
    # One request: the local header (30 bytes plus name and extra fields, which the
    # central directory sizes only approximately) and the start of the data.
    slack = 30 + len(info.filename.encode()) + 1024
    span = min(info.compress_size, max(n, 1 << 16))
    block = fetch_range(url, info.header_offset, info.header_offset + slack + span - 1)
    name_len, extra_len = struct.unpack('<HH', block[26:30])
    data = block[30 + name_len + extra_len:][:span]
    if info.compress_type == zipfile.ZIP_STORED:
        return data[:n]
    return zlib.decompressobj(-15).decompress(data, n)


def open_archive(url):
    return zipfile.ZipFile(RangeFile(url))


def extract(archive, member, path):
    """Stream one member to path. zipfile verifies the member's CRC-32 on the way."""
    with archive.open(member) as source, open(path, 'wb') as out:
        while chunk := source.read(1 << 22):
            out.write(chunk)


def fetch_member(url, spec):
    """Bytes of one member from its recorded offset and sizes: two range requests,
    no central directory. spec holds offset, compressed_size and compress_type."""
    head = fetch_range(url, spec['offset'], spec['offset'] + 29)
    if head[:4] != b'PK\x03\x04':
        raise OSError(f"No local ZIP header at offset {spec['offset']}")
    name_len, extra_len = struct.unpack('<HH', head[26:30])
    start = spec['offset'] + 30 + name_len + extra_len
    data = fetch_range(url, start, start + spec['compressed_size'] - 1)
    if spec['compress_type'] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif spec['compress_type'] != zipfile.ZIP_STORED:
        raise OSError(f"Unsupported ZIP compression {spec['compress_type']}")
    return data
