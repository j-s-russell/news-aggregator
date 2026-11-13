import re
from urllib.parse import urlparse


SPORTS_DOMAINS = ('nbcsports.com', 'foxsports.com', 'cbssports.com', 'espn.com', 'si.com', 'theathletic.com')

# League acronyms are matched case-sensitively so words like "nba" in other contexts don't hit
_SPORTS_ACRONYMS = re.compile(r'\b(NFL|NBA|WNBA|MLB|NHL|MLS|NCAA|PGA|UFC|EFL)\b')
_SPORTS_PHRASES = re.compile(
    r'\b(super bowl|world series|stanley cup|march madness|premier league|champions league|'
    r'world cup|playoffs?|quarterback|touchdown|grand slam|wimbledon|soccer|jockey|horse ?race|'
    r'(?<!political )football)\b',
    re.IGNORECASE,
)


def _as_str(value):
    return value if isinstance(value, str) else ''


def is_sports_url(url):
    """True for sports URLs, e.g. USA Today team sites like steelerswire.usatoday.com."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if any(host == d or host.endswith('.' + d) for d in SPORTS_DOMAINS):
        return True
    if host.endswith('usatoday.com') and host != 'www.usatoday.com':
        return True
    path = parsed.path.lower()
    # OutKick is Fox's sports/betting brand (/outkick-sports/, /outkick-betting/, ...)
    if path.startswith('/outkick'):
        return True
    # Matches /sport/ (BBC, CNN) and /sports/ (CBS, Breitbart, USA Today, Fox, WaPo, ABC)
    return re.search(r'/([a-z]+-)?sports?(/|$)', path) is not None


def is_sports_text(title, description=None):
    """Keyword fallback for sources whose URLs carry no section (Newsweek, HuffPost, ...)."""
    text = f"{_as_str(title)} {_as_str(description)}"
    return bool(_SPORTS_ACRONYMS.search(text) or _SPORTS_PHRASES.search(text))


def is_sports(row):
    """Row-wise check for a DataFrame row (or dict) with url/title/description."""
    return is_sports_url(row.get('url')) or is_sports_text(row.get('title'), row.get('description'))
