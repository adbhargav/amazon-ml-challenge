"""Static lexicons used by the normalisers.

Everything is keyed by *country string* as it appears in the data.  Unknown
countries simply fall back to the generic (empty) tables, so the pipeline runs
unchanged for a country never seen before (France in the test set is handled
explicitly here, but the design does not require it).
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Legal-form suffixes -> canonical token.  Dotted / spaced forms are collapsed
# by the tokenizer first (l.l.c -> llc, s a r l -> sarl).
# ---------------------------------------------------------------------------
SUFFIX_CANON = {
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "company": "co", "compagnie": "co", "cie": "co", "co": "co",
    "limited": "ltd", "ltd": "ltd", "ltee": "ltd",
    "private": "pvt", "pvt": "pvt",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp", "plc": "plc",
    "pc": "pc", "pa": "pa", "opc": "opc",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci",
    "snc": "snc", "eurl": "eurl", "sca": "sca", "scp": "scp", "scop": "scop",
    "sem": "sem", "gie": "gie",
    "trust": "trust", "foundation": "foundation", "association": "association",
    "assn": "association", "assoc": "association",
    "enterprises": "enterprises", "enterprise": "enterprises",
    "industries": "industries", "holdings": "holdings", "holding": "holdings",
    "group": "group", "groupe": "group",
}

# Tokens that carry no entity identity (legal form).  Removed from the "core".
LEGAL_TOKENS = {
    "corp", "inc", "co", "ltd", "pvt", "llc", "pllc", "llp", "lp", "plc", "pc",
    "pa", "opc", "gmbh", "ag", "bv", "nv", "sarl", "sas", "sasu", "sa", "sci",
    "snc", "eurl", "sca", "scp", "scop", "sem", "gie",
}

STOP_TOKENS = {
    "and", "of", "the", "a", "an",
    "de", "des", "du", "la", "le", "les", "d", "l", "et", "en", "au", "aux",
}

# Alias markers: split a name into variants around these.
ALIAS_RE = re.compile(
    r"\s+(?:d\.?\s?b\.?\s?a\.?|a\.?k\.?a\.?|t/a|trading\s+as|f/k/a|fka|formerly)\s+",
    re.IGNORECASE,
)
MS_PREFIX_RE = re.compile(r"^\s*m/s\.?\s+", re.IGNORECASE)
JUNK_PREFIX_RE = re.compile(r"^[\W_]+")
JUNK_SUFFIX_RE = re.compile(r"[\s\-\*\#\>\<\.\,\|\"\']+$")
DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|org|net|co|io|biz|fr|co\.in|co\.uk)$"
)
LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})
ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$")

# ---------------------------------------------------------------------------
# Street-type abbreviations (country gated).  Keys are the abbreviated forms
# after lowercasing and punctuation removal; values the canonical word.
# ---------------------------------------------------------------------------
STREET_ABBR_US = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "blv": "boulevard", "ln": "lane", "ct": "court",
    "pl": "place", "cir": "circle", "pkwy": "parkway", "pky": "parkway", "hwy": "highway",
    "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square", "expy": "expressway",
    "fwy": "freeway", "mt": "mount", "ft": "fort", "hts": "heights", "jct": "junction",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "ste": "suite", "apt": "apartment", "bldg": "building", "fl": "floor",
    "rte": "route", "crk": "creek", "xing": "crossing", "plz": "plaza",
}
STREET_ABBR_IN = {
    "rd": "road", "st": "street", "nr": "near", "opp": "opposite",
    "sec": "sector", "sect": "sector", "blk": "block", "bldg": "building",
    "apt": "apartment", "apts": "apartments", "flr": "floor", "fl": "floor",
    "gnd": "ground", "ind": "industrial", "indl": "industrial", "estt": "estate",
    "ext": "extension", "extn": "extension", "colony": "colony", "ngr": "nagar",
    "vlg": "village", "vill": "village", "dist": "district", "distt": "district",
    "tal": "taluka", "tq": "taluka", "teh": "tehsil", "ph": "phase", "hno": "",
    "no": "", "nos": "", "h": "", "num": "",
}
STREET_ABBR_FR = {
    "r": "rue", "rte": "route", "av": "avenue", "ave": "avenue", "bd": "boulevard",
    "boul": "boulevard", "bvd": "boulevard", "imp": "impasse", "all": "allee",
    "ch": "chemin", "chem": "chemin", "pl": "place", "crs": "cours", "qu": "quai",
    "sq": "square", "st": "saint", "ste": "sainte", "res": "residence", "lot": "lotissement",
    "zi": "zone industrielle", "za": "zone artisanale", "zac": "zac", "cdx": "cedex",
    "bat": "batiment", "esc": "escalier", "et": "etage", "n": "", "no": "",
}
STREET_ABBR = {"US": STREET_ABBR_US, "India": STREET_ABBR_IN, "France": STREET_ABBR_FR}

# Number markers that precede a house / plot / door number.
NUMBER_MARKER_RE = re.compile(
    r"\b(?:h\.?\s?no\.?|hno|house\s*no\.?|door\s*no\.?|plot\s*no\.?|flat\s*no\.?|shop\s*no\.?|"
    r"kh\.?\s*no\.?|khasra\s*no\.?|sy\.?\s*no\.?|survey\s*no\.?|no\.?|n[°º]|#)\s*[-:]?\s*",
    re.IGNORECASE,
)
UNIT_RE = re.compile(
    r"\b(?:unit|apt|apartment|suite|ste|fl|floor|flat|room|rm|bldg|building|office|shop)\s*#?\s*([a-z0-9\-/]+)",
    re.IGNORECASE,
)
LANDMARK_RE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next\s+to|adjacent|above|below|in\s+front\s+of)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# State / region lexicons: normalised component string -> canonical state code
# ---------------------------------------------------------------------------
_US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new hampshire",
    "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington",
    "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
    "PR": "puerto rico", "GU": "guam", "VI": "virgin islands",
}
_IN_STATES = {
    "AP": "andhra pradesh", "AR": "arunachal pradesh", "AS": "assam", "BR": "bihar",
    "CG": "chhattisgarh", "GA": "goa", "GJ": "gujarat", "HR": "haryana", "HP": "himachal pradesh",
    "JH": "jharkhand", "KA": "karnataka", "KL": "kerala", "MP": "madhya pradesh",
    "MH": "maharashtra", "MN": "manipur", "ML": "meghalaya", "MZ": "mizoram", "NL": "nagaland",
    "OD": "odisha", "PB": "punjab", "RJ": "rajasthan", "SK": "sikkim", "TN": "tamil nadu",
    "TS": "telangana", "TR": "tripura", "UP": "uttar pradesh", "UK": "uttarakhand",
    "WB": "west bengal", "DL": "delhi", "CH": "chandigarh", "JK": "jammu and kashmir",
    "LA": "ladakh", "PY": "puducherry", "AN": "andaman and nicobar islands",
    "DN": "dadra and nagar haveli and daman and diu", "LD": "lakshadweep",
}
_IN_STATE_ALIASES = {
    "orissa": "OD", "odisha": "OD", "uttaranchal": "UK", "pondicherry": "PY", "new delhi": "DL",
    "delhi ncr": "DL", "nct of delhi": "DL", "chattisgarh": "CG", "telengana": "TS",
    "tamilnadu": "TN", "karnatka": "KA", "maharastra": "MH", "gujrat": "GJ", "kerela": "KL",
    "west bengal": "WB", "bengal": "WB", "andhra": "AP", "up": "UP", "mp": "MP", "hp": "HP",
    "j&k": "JK", "j and k": "JK", "daman and diu": "DN", "dadra and nagar haveli": "DN",
}
# France: region -> canonical code; department (name or number) -> region code.
_FR_REGIONS = {
    "ARA": "auvergne-rhone-alpes", "BFC": "bourgogne-franche-comte", "BRE": "bretagne",
    "CVL": "centre-val de loire", "COR": "corse", "GES": "grand est", "HDF": "hauts-de-france",
    "IDF": "ile-de-france", "NOR": "normandie", "NAQ": "nouvelle-aquitaine", "OCC": "occitanie",
    "PDL": "pays de la loire", "PAC": "provence-alpes-cote d'azur",
}
_FR_REGION_ALIASES = {
    "auvergne rhone alpes": "ARA", "rhone alpes": "ARA", "auvergne": "ARA",
    "bourgogne franche comte": "BFC", "bourgogne": "BFC", "franche comte": "BFC",
    "bretagne": "BRE", "brittany": "BRE", "centre val de loire": "CVL", "centre": "CVL",
    "corse": "COR", "corsica": "COR", "grand est": "GES", "alsace": "GES", "lorraine": "GES",
    "champagne ardenne": "GES", "hauts de france": "HDF", "nord pas de calais": "HDF",
    "picardie": "HDF", "ile de france": "IDF", "normandie": "NOR", "normandy": "NOR",
    "nouvelle aquitaine": "NAQ", "aquitaine": "NAQ", "limousin": "NAQ", "poitou charentes": "NAQ",
    "occitanie": "OCC", "midi pyrenees": "OCC", "languedoc roussillon": "OCC",
    "pays de la loire": "PDL", "provence alpes cote d azur": "PAC", "paca": "PAC", "provence": "PAC",
}
_FR_DEPARTMENTS = {  # number: (name, region)
    "01": ("ain", "ARA"), "02": ("aisne", "HDF"), "03": ("allier", "ARA"),
    "04": ("alpes de haute provence", "PAC"), "05": ("hautes alpes", "PAC"),
    "06": ("alpes maritimes", "PAC"), "07": ("ardeche", "ARA"), "08": ("ardennes", "GES"),
    "09": ("ariege", "OCC"), "10": ("aube", "GES"), "11": ("aude", "OCC"), "12": ("aveyron", "OCC"),
    "13": ("bouches du rhone", "PAC"), "14": ("calvados", "NOR"), "15": ("cantal", "ARA"),
    "16": ("charente", "NAQ"), "17": ("charente maritime", "NAQ"), "18": ("cher", "CVL"),
    "19": ("correze", "NAQ"), "21": ("cote d or", "BFC"), "22": ("cotes d armor", "BRE"),
    "23": ("creuse", "NAQ"), "24": ("dordogne", "NAQ"), "25": ("doubs", "BFC"), "26": ("drome", "ARA"),
    "27": ("eure", "NOR"), "28": ("eure et loir", "CVL"), "29": ("finistere", "BRE"),
    "2a": ("corse du sud", "COR"), "2b": ("haute corse", "COR"), "30": ("gard", "OCC"),
    "31": ("haute garonne", "OCC"), "32": ("gers", "OCC"), "33": ("gironde", "NAQ"),
    "34": ("herault", "OCC"), "35": ("ille et vilaine", "BRE"), "36": ("indre", "CVL"),
    "37": ("indre et loire", "CVL"), "38": ("isere", "ARA"), "39": ("jura", "BFC"),
    "40": ("landes", "NAQ"), "41": ("loir et cher", "CVL"), "42": ("loire", "ARA"),
    "43": ("haute loire", "ARA"), "44": ("loire atlantique", "PDL"), "45": ("loiret", "CVL"),
    "46": ("lot", "OCC"), "47": ("lot et garonne", "NAQ"), "48": ("lozere", "OCC"),
    "49": ("maine et loire", "PDL"), "50": ("manche", "NOR"), "51": ("marne", "GES"),
    "52": ("haute marne", "GES"), "53": ("mayenne", "PDL"), "54": ("meurthe et moselle", "GES"),
    "55": ("meuse", "GES"), "56": ("morbihan", "BRE"), "57": ("moselle", "GES"), "58": ("nievre", "BFC"),
    "59": ("nord", "HDF"), "60": ("oise", "HDF"), "61": ("orne", "NOR"), "62": ("pas de calais", "HDF"),
    "63": ("puy de dome", "ARA"), "64": ("pyrenees atlantiques", "NAQ"), "65": ("hautes pyrenees", "OCC"),
    "66": ("pyrenees orientales", "OCC"), "67": ("bas rhin", "GES"), "68": ("haut rhin", "GES"),
    "69": ("rhone", "ARA"), "70": ("haute saone", "BFC"), "71": ("saone et loire", "BFC"),
    "72": ("sarthe", "PDL"), "73": ("savoie", "ARA"), "74": ("haute savoie", "ARA"), "75": ("paris", "IDF"),
    "76": ("seine maritime", "NOR"), "77": ("seine et marne", "IDF"), "78": ("yvelines", "IDF"),
    "79": ("deux sevres", "NAQ"), "80": ("somme", "HDF"), "81": ("tarn", "OCC"), "82": ("tarn et garonne", "OCC"),
    "83": ("var", "PAC"), "84": ("vaucluse", "PAC"), "85": ("vendee", "PDL"), "86": ("vienne", "NAQ"),
    "87": ("haute vienne", "NAQ"), "88": ("vosges", "GES"), "89": ("yonne", "BFC"),
    "90": ("territoire de belfort", "BFC"), "91": ("essonne", "IDF"), "92": ("hauts de seine", "IDF"),
    "93": ("seine saint denis", "IDF"), "94": ("val de marne", "IDF"), "95": ("val d oise", "IDF"),
}


def _norm_key(s: str) -> str:
    """Lowercase, strip accents/punctuation -> single-spaced ascii for lexicon lookups."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def _build_state_lex() -> dict[str, dict[str, str]]:
    lex: dict[str, dict[str, str]] = {"US": {}, "India": {}, "France": {}}
    for code, name in _US_STATES.items():
        lex["US"][code.lower()] = code
        lex["US"][_norm_key(name)] = code
    for code, name in _IN_STATES.items():
        lex["India"][code.lower()] = code
        lex["India"][_norm_key(name)] = code
    for alias, code in _IN_STATE_ALIASES.items():
        lex["India"][_norm_key(alias)] = code
    for code, name in _FR_REGIONS.items():
        lex["France"][_norm_key(name)] = code
    for alias, code in _FR_REGION_ALIASES.items():
        lex["France"][_norm_key(alias)] = code
    for num, (name, code) in _FR_DEPARTMENTS.items():
        lex["France"][_norm_key(name)] = code
        lex["France"][f"dept {num}"] = code
    return lex


STATE_LEX = _build_state_lex()

# postcode patterns per country (generic fallback: 4-6 digits)
POSTCODE_RE = {
    "US": re.compile(r"^\d{5}(?:-\d{4})?$"),
    "India": re.compile(r"^\d{6}$"),
    "France": re.compile(r"^\d{5}$"),
}
POSTCODE_GENERIC_RE = re.compile(r"^\d{4,6}$")

CITY_AFFIX_RE = re.compile(
    r"^(?:city of|town of|village of|borough of|township of)\s+|\s+(?:city|town|cdp|township|village|borough|municipality|nagar|taluka|tehsil|district|cedex\s*\d*)$"
)


def street_abbr(country: str) -> dict[str, str]:
    return STREET_ABBR.get(country, {})


def state_lex(country: str) -> dict[str, str]:
    return STATE_LEX.get(country, {})


def postcode_re(country: str) -> re.Pattern:
    return POSTCODE_RE.get(country, POSTCODE_GENERIC_RE)
