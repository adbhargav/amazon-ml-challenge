"""Synthetic data generator that reproduces the challenge's file format and
its documented noise catalogue.

It exists so the pipeline can be developed, unit-tested and benchmarked
end-to-end on a machine that does not have the real data.  Statistics are
taken from the public analyses of the real training set:

* 5.6% singletons, 0-11 matches per S1 (mode 3-4), ~26% of pool records are
  distractors, half of the S1 names are re-used by another S1 in a different
  place, 70% of singletons have a same-name look-alike in the pool;
* name noise: suffix abbreviation / dropping, word-order shuffle, typos,
  accents, l33t digits, domain names, DBA prefixes, junk prefixes, Indic
  transliteration (India), mixed script;
* address noise: abbreviations, component re-ordering / dropping, "null"
  components, empty addresses, house-number noise, landmark references,
  state code <-> name, department instead of region (France).

The Indic "script" is a deterministic letter mapping to Devanagari code
points: linguistically meaningless, but it is *consistent per word*, which is
exactly the property the learned transliteration dictionary relies on.
"""
from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- vocab
_WORDS_EN = """summit health metro cardiology medical centre clinic silver foundation blue management raj investments
super software elevate impex perfect technology pali energy associates meyer garrett auk pono cornerstone
indo sky tucker transalta pantages payne enterprises womens united care shreyans holdings all logistics
srp advisors hyderabad global systems pacific coast fresh harvest golden gate prime motors river valley
sunrise bakery apex consulting delta freight north star pioneer builders green leaf organic quantum labs
crescent trading oak tree dental bright future learning iron bridge steel royal palace hotel city
lakeside marina mountain view ranch atlas security phoenix media eagle eye optics jade garden restaurant
liberty tax services harbor point realty maple grove nursing horizon travel cobalt mining
vertex analytics nimbus cloud pixel studio orbit telecom velvet fashion crystal water works zenith pharma
alpha beta gamma omega sigma nova terra luna sol vista mesa aurora""".split()
_WORDS_FR = """boulangerie patisserie garage carrosserie menuiserie plomberie electricite conseil groupe
fils freres atelier maison cabinet pharmacie optique coiffure restaurant bistro cave transport
nettoyage renovation immobilier gestion services etudes marine ecole club amicale fractales thermal grain
znb dunkerque lille bordeaux nantes roubaix perso batiment travaux jardin fleurs boucherie""".split()

_SUFFIX = {
    "US": ["LLC", "Inc", "Corp", "Corporation", "Incorporated", "Co", "Company", "LLP", "PLLC", "Ltd", ""],
    "India": ["Private Limited", "Pvt Ltd", "Pvt. Ltd.", "Limited", "LLP", "OPC", "Enterprises", ""],
    "France": ["SARL", "SAS", "SASU", "SA", "SCI", "EURL", "S.A.S", "S.A.R.L", ""],
}
_SUFFIX_VARIANTS = {
    "llc": ["LLC", "L.L.C.", "L L C", "Llc"], "inc": ["Inc", "Inc.", "Incorporated"],
    "corp": ["Corp", "Corp.", "Corporation"], "co": ["Co", "Co.", "Company"],
    "private limited": ["Private Limited", "Pvt Ltd", "Pvt. Ltd.", "Pvt Limited", "Private Ltd", "P Ltd"],
    "limited": ["Limited", "Ltd", "Ltd."], "sarl": ["SARL", "S.A.R.L", "S.A.R.L."],
    "sas": ["SAS", "S.A.S", "S.A.S."], "sasu": ["SASU", "S.A.S.U"],
}

_CITIES = {
    "US": [("Euclid", "OH", "44117"), ("Columbus", "OH", "43004"), ("Peoria", "IL", "61602"),
           ("Kansas City", "MO", "64101"), ("Tullahoma", "TN", "37388"), ("Austin", "TX", "78701"),
           ("Denver", "CO", "80202"), ("Tampa", "FL", "33602"), ("Seattle", "WA", "98101"),
           ("Newark", "NJ", "07102"), ("Boise", "ID", "83702"), ("Mesa", "AZ", "85201")],
    "India": [("Hyderabad", "Telangana", "500001"), ("Lucknow", "Uttar Pradesh", "226001"),
              ("New Delhi", "Delhi", "110001"), ("Mumbai", "Maharashtra", "400001"),
              ("Bangalore", "Karnataka", "560001"), ("Chennai", "Tamil Nadu", "600001"),
              ("Pune", "Maharashtra", "411001"), ("Ahmedabad", "Gujarat", "380001"),
              ("Jaipur", "Rajasthan", "302001"), ("Kolkata", "West Bengal", "700001")],
    "France": [("Lille", "Hauts-de-France", "59000"), ("Dunkerque", "Hauts-de-France", "59140"),
               ("Roubaix", "Hauts-de-France", "59100"), ("Bordeaux", "Nouvelle-Aquitaine", "33000"),
               ("La Teste-de-Buch", "Nouvelle-Aquitaine", "33260"), ("Nantes", "Pays de la Loire", "44000"),
               ("Saint-Nazaire", "Pays de la Loire", "44600"), ("Amiens", "Hauts-de-France", "80000")],
}
_STATE_CODE = {"Telangana": "TS", "Uttar Pradesh": "UP", "Delhi": "DL", "Maharashtra": "MH", "Karnataka": "KA",
               "Tamil Nadu": "TN", "Gujarat": "GJ", "Rajasthan": "RJ", "West Bengal": "WB"}
_STATE_NAME = {"OH": "Ohio", "IL": "Illinois", "MO": "Missouri", "TN": "Tennessee", "TX": "Texas", "CO": "Colorado",
               "FL": "Florida", "WA": "Washington", "NJ": "New Jersey", "ID": "Idaho", "AZ": "Arizona"}
_FR_DEPT = {"Lille": "Nord", "Dunkerque": "Nord", "Roubaix": "Nord", "Bordeaux": "Gironde", "La Teste-de-Buch": "Gironde",
            "Nantes": "Loire-Atlantique", "Saint-Nazaire": "Loire-Atlantique", "Amiens": "Somme"}

_STREETS = {
    "US": [("Euclid", "Avenue"), ("Ravens Glen", "Drive"), ("Fremont", "Street"), ("45th", "Terrace"), ("Main", "Street"),
           ("Oak", "Lane"), ("Broadway", ""), ("Cedar", "Boulevard"), ("Lake Shore", "Drive"), ("Hill", "Road"), ("Park", "Place"),
           ("Maple", "Court"), ("Washington", "Avenue"), ("Elm", "Street"), ("Sunset", "Boulevard")],
    "India": [("MG", "Road"), ("Banjara Hills", ""), ("Sector 18", ""), ("Nehru", "Marg"), ("Station", "Road"),
              ("Gandhi", "Nagar"), ("Anna", "Salai"), ("Link", "Road"), ("Ring", "Road"), ("Industrial Estate", ""),
              ("Asha Complex", ""), ("Jubilee Hills", ""), ("Civil Lines", "")],
    "France": [("Rue", "Pierre Dignac"), ("Boulevard", "du President Franklin Roosevelt"), ("Rue", "Parmentier"),
               ("Avenue", "de Dunkerque"), ("Rue", "de Dieppe"), ("Rue", "Jean Zay"), ("Rue", "Jules Andrieu"),
               ("Rue", "de Lannoy"), ("Rue", "des Girondins"), ("Rue", "Mestrezat"), ("Place", "de la Gare"),
               ("Chemin", "des Ecoles"), ("Allee", "des Tilleuls"), ("Impasse", "du Moulin")],
}
_ABBR = {"Avenue": "Ave", "Drive": "Dr", "Street": "St", "Terrace": "Ter", "Boulevard": "Blvd", "Lane": "Ln",
         "Road": "Rd", "Court": "Ct", "Place": "Pl", "Rue": "R.", "Chemin": "Ch.", "Allee": "All.", "Impasse": "Imp."}
_ACCENTS = {"a": "á", "e": "é", "o": "ó", "u": "ú", "i": "í", "E": "É", "A": "Á"}
_LEET = {"o": "0", "l": "1", "e": "3", "a": "4", "s": "5", "t": "7"}
_DEVA = {c: chr(0x0915 + i) for i, c in enumerate(string.ascii_lowercase)}   # consistent fake script
_TAMIL = {c: chr(0x0B95 + i) for i, c in enumerate(string.ascii_lowercase)}


def _to_script(word: str, table: Dict[str, str]) -> str:
    return "".join(table.get(ch.lower(), ch) for ch in word)


class Generator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._next_id = 10_000_000 + self.rng.randrange(1_000_000)

    # ----------------------------------------------------------- helpers
    def new_id(self, prefix: str) -> str:
        self._next_id += self.rng.randrange(1, 40)
        return f"{prefix}-{self._next_id}"

    def name_core(self, country: str) -> List[str]:
        words = _WORDS_FR if country == "France" else _WORDS_EN
        k = self.rng.choice([1, 2, 2, 2, 3, 3, 4])
        return [self.rng.choice(words).capitalize() for _ in range(k)]

    def address(self, country: str) -> dict:
        city, state, pc = self.rng.choice(_CITIES[country])
        a, b = self.rng.choice(_STREETS[country])
        hn = str(self.rng.randrange(1, 9999))
        return {"hn": hn, "street": (a, b), "city": city, "state": state, "pc": pc}

    @staticmethod
    def fmt_address(a: dict, country: str) -> str:
        sa, sb = a["street"]
        street = f"{sa} {sb}".strip() if country != "France" else f"{sa} {sb}"
        if country == "US":
            return f"{a['hn']} {street}, {a['city']}, {a['state']}"
        if country == "India":
            return f"{a['hn']}, {street}, {a['city']}, {a['state']}"
        return f"{a['hn']} {street}, {a['city']}, {a['state']}"

    # ------------------------------------------------------------ noise
    def noisy_name(self, core: List[str], suffix: str, country: str) -> str:
        r = self.rng
        core = list(core)
        ops = r.random()
        # suffix handling
        sfx = suffix
        if suffix and r.random() < 0.35:
            sfx = ""                                          # truncated
        elif suffix and r.random() < 0.5:
            key = suffix.lower().replace(".", "").strip()
            if key in _SUFFIX_VARIANTS:
                sfx = r.choice(_SUFFIX_VARIANTS[key])
        if r.random() < 0.18 and len(core) > 1:              # word order shuffle
            r.shuffle(core)
        if r.random() < 0.10:                                 # extra word
            core.append(r.choice(["Group", "Services", "Solutions", "Global", "Corp"]))
        if r.random() < 0.14:                                 # accents
            core = [("".join(_ACCENTS.get(ch, ch) if r.random() < 0.3 else ch for ch in w)) for w in core]
        if r.random() < 0.22:                                 # typos
            w = r.randrange(len(core))
            s = list(core[w])
            for _ in range(r.choice([1, 1, 2, 3])):
                if len(s) > 2:
                    i = r.randrange(len(s))
                    op = r.random()
                    if op < 0.4:
                        s[i] = r.choice(string.ascii_lowercase)
                    elif op < 0.7:
                        del s[i]
                    else:
                        s.insert(i, r.choice(string.ascii_lowercase))
            core[w] = "".join(s)
        if r.random() < 0.03:                                 # l33t
            core = ["".join(_LEET.get(ch, ch) if r.random() < 0.4 else ch for ch in w) for w in core]
        name = " ".join(core + ([sfx] if sfx else []))
        if r.random() < 0.06:                                 # domain name
            base = "".join(w.lower() for w in core)
            base = "".join(ch for ch in base if ch.isalnum())
            if r.random() < 0.2:
                base = "".join(_LEET.get(ch, ch) if r.random() < 0.2 else ch for ch in base)
            name = base + ".com"
        if country == "India" and r.random() < 0.17:          # transliteration
            table = _DEVA if r.random() < 0.75 else _TAMIL
            words = name.split()
            if r.random() < 0.05:                              # mixed script
                words = [(_to_script(w, table) if r.random() < 0.5 else w) for w in words]
            else:
                words = [_to_script(w, table) for w in words]
            name = " ".join(words)
        if r.random() < 0.04:                                 # dba prefix
            name = f"{r.choice(_WORDS_EN).capitalize()}{r.choice(_WORDS_EN)} dba {name}"
        if r.random() < 0.01:
            name = r.choice(["-- ", "*** ", ">> ", "# "]) + name
        if r.random() < 0.3:
            name = name.upper()
        return name

    def noisy_address(self, a: dict, country: str) -> str:
        r = self.rng
        if r.random() < 0.035:
            return ""
        hn = a["hn"]
        if r.random() < 0.08:
            hn = "0" * r.choice([1, 2]) + hn
        elif r.random() < 0.04:
            hn = str(int(hn) + r.choice([-1, 1, 4]))
        elif r.random() < 0.03 and len(hn) > 2:
            hn = hn[:-1]
        sa, sb = a["street"]
        if r.random() < 0.45:
            sa, sb = _ABBR.get(sa, sa), _ABBR.get(sb, sb)
        street = f"{sa} {sb}".strip()
        state = a["state"]
        if country == "US" and r.random() < 0.2:
            state = _STATE_NAME.get(state, state)
        if country == "India" and r.random() < 0.25:
            state = _STATE_CODE.get(state, state)
        if country == "France" and r.random() < 0.4:
            state = _FR_DEPT.get(a["city"], state)
        comps = []
        if country == "India" and r.random() < 0.15:
            comps.append(r.choice(["Near SBI ATM", "Opp Bus Stand", "Behind Temple", "Near Metro Station"]))
        if r.random() < 0.18:                                 # drop house number
            comps.append(street)
        else:
            comps.append(f"{hn} {street}" if country != "India" else f"{r.choice(['', 'No. ', 'H.No ', 'Plot No '])}{hn}, {street}")
        if r.random() < 0.85:
            comps.append(a["city"] if r.random() > 0.05 else a["city"].upper())
        if r.random() < 0.75:
            comps.append(state)
        if r.random() < 0.25:
            comps.append(a["pc"])
        if r.random() < 0.02:
            comps.insert(1, "null")
        if r.random() < 0.3:
            r.shuffle(comps)
        s = ", ".join(c for c in comps if c)
        if r.random() < 0.35:
            s = s.upper()
        return s

    # ------------------------------------------------------------ build
    def build(self, n_s1: int, country_mix: Dict[str, float], with_gt: bool) -> Tuple[List, List, List, List]:
        r = self.rng
        countries = list(country_mix)
        weights = [country_mix[c] for c in countries]
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        name_pool: Dict[str, List[Tuple[List[str], str]]] = {c: [] for c in countries}
        match_dist = [(0, 5.6), (1, 5.4), (2, 17), (3, 24), (4, 22), (5, 14.6), (6, 7.5), (7, 2.9), (8, 0.9), (9, 0.2)]
        ks, ws = zip(*match_dist)

        entities = []
        for _ in range(n_s1):
            c = r.choices(countries, weights)[0]
            if name_pool[c] and r.random() < 0.5:             # re-use a name (different place)
                core, suffix = r.choice(name_pool[c])
            else:
                core, suffix = self.name_core(c), r.choice(_SUFFIX[c])
                name_pool[c].append((core, suffix))
            addr = self.address(c)
            entities.append((c, core, suffix, addr))

        def emit_pool(c, core, suffix, addr):
            src = r.choice([2, 3])
            eid = self.new_id(f"S{src}")
            row = (eid, self.noisy_name(core, suffix, c), self.noisy_address(addr, c), c)
            (s2_rows if src == 2 else s3_rows).append(row)
            return eid

        for c, core, suffix, addr in entities:
            sid = self.new_id("S1")
            s1_rows.append((sid, " ".join(core + ([suffix] if suffix else [])), self.fmt_address(addr, c), c))
            k = r.choices(ks, ws)[0]
            matched = [emit_pool(c, core, suffix, addr) for _ in range(k)]
            gt_rows.append((sid, ",".join(matched)))
            # distractors: same-name twin elsewhere (esp. for singletons) or a random entity
            n_twin = (1 if r.random() < 0.7 else 0) if k == 0 else (1 if r.random() < 0.22 else 0)
            for _ in range(n_twin):
                twin_addr = self.address(c)
                if r.random() < 0.15:                          # same street, nearby number
                    twin_addr = dict(addr, hn=str(int(addr["hn"]) + r.choice([2, 4, -3])))
                emit_pool(c, core, suffix, twin_addr)
            if r.random() < 0.12:
                emit_pool(c, self.name_core(c), r.choice(_SUFFIX[c]), self.address(c))
        r.shuffle(s2_rows)
        r.shuffle(s3_rows)
        return s1_rows, s2_rows, s3_rows, (gt_rows if with_gt else [])


def _write_tsv(path: Path, header: List[str], rows: List[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def write_dataset(out_dir: str | Path, n_train: int = 3000, n_test: int = 2000, seed: int = 0) -> None:
    out = Path(out_dir)
    g = Generator(seed)
    tr = g.build(n_train, {"US": 0.6, "India": 0.4}, with_gt=True)
    te = g.build(n_test, {"US": 0.38, "India": 0.47, "France": 0.15}, with_gt=True)
    hdr = ["entity_id", "business_name", "business_address", "country"]
    for split, (s1, s2, s3, gt) in (("train", tr), ("test", te)):
        d = out / split
        _write_tsv(d / f"{split}_source1.tsv", hdr, s1)
        _write_tsv(d / f"{split}_source2.tsv", hdr, s2)
        _write_tsv(d / f"{split}_source3.tsv", hdr, s3)
        if split == "train":
            _write_tsv(d / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
        else:  # kept outside the official layout so the pipeline never sees it
            _write_tsv(out / "test_ground_truth_HIDDEN.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
