from ber.normalize_address import build_city_lexicon, parse_address
from ber.normalize_name import WordSegmenter, normalize_name
from ber.translit import Translit

TL = Translit()
SEG = WordSegmenter.from_texts(["pcl medical centre", "elevate impex", "tullahoma youth fellowship"] * 3, min_count=2)


def name(s):
    return normalize_name(s, TL, SEG)


def test_suffix_canonical_and_core():
    r = name("AUK Pono L.L.C.")
    assert r["n_core"] == "auk pono" and r["n_suffix"] == "llc"
    assert name("Perfect Technology Private Limited")["n_suffix"] == "ltd pvt"
    assert name("Perfect Technology Pvt. Ltd.")["n_key"] == name("Perfect Technology Private Limited")["n_key"]


def test_junk_prefix_ampersand_accents_leet():
    assert name("-- Meyer & Garrett")["n_core"] == "meyer garrett"
    assert name("Tucker Metro Tránsalta")["n_core"] == "tucker metro transalta"
    r = name("Indo 5ky Energy")
    assert r["n_core"] == "indo sky energy" and r["n_raw_core"] == "indo 5ky energy"
    assert name("007 Costamare LLC")["n_core"] == "007 costamare"  # pure digits untouched


def test_domain_segmentation():
    r = name("pclmedicalcentre.com")
    assert r["n_is_domain"] and r["n_core"] == "pcl medical centre" and r["n_concat"] == "pclmedicalcentre"
    r = name("*** tullah0mayouthfellowship.com")
    assert r["n_is_domain"] and r["n_core"] == "tullahoma youth fellowship"


def test_alias_variants():
    r = name("Tavozephdelta dba Perfect Technology Pvt Ltd")
    assert r["n_has_alias"]
    assert r["n_alt2"] == "perfect technology" and r["n_alt1"] == "tavozephdelta"
    assert "dba" not in r["n_core"]


def test_transliteration_dictionary():
    tl = Translit.fit([("Silver Foundation Private Limited", "सिल्वर फाउंडेशन प्राइवेट लिमिटेड")] * 3
                      + [("Silver Foundation Pvt Ltd", "सिल्वर फाउंडेशन प्रा. लिमिटेड")] * 2)
    assert tl.table["सिल्वर"] == "silver" and tl.table["प्रा"] == "pvt"
    r = normalize_name("सिल्वर फाउंडेशन प्राइवेट लिमिटेड", tl, SEG)
    assert r["n_core"] == "silver foundation" and r["n_suffix"] == "ltd pvt" and r["n_script"] == 1 and r["n_oov"] == 0
    r = normalize_name("Sun कंसल्टेंसी Private Limited", tl, SEG)
    assert r["n_script"] == 2 and r["n_oov"] == 1 and r["n_core"].startswith("sun ")


CITIES = {"US": {"euclid", "columbus", "peoria", "new york", "kansas"},
          "France": {"lille", "bordeaux", "roubaix", "nantes", "la teste de buch", "dunkerque"},
          "India": {"lucknow", "bangalore", "west delhi"}}


def addr(a, c):
    return parse_address(a, c, TL, CITIES)


def test_us_reordered_components():
    a = addr("24800 Euclid Avenue, Euclid, OH", "US")
    b = addr("OH, EUCLID AVE, EUCLID", "US")
    assert a["a_state"] == b["a_state"] == "OH"
    assert a["a_city"] == b["a_city"] == "euclid"
    assert a["a_street"] == b["a_street"] == "euclid avenue"
    assert a["a_hn"] == "24800" and a["a_postcode"] == "" and b["a_hn"] == ""


def test_null_and_empty():
    a = addr("3315 FREMONT ST, null, PEORIA, IL", "US")
    assert a["a_city"] == "peoria" and a["a_hn"] == "3315" and a["a_street"] == "fremont street"
    assert addr("", "US")["a_empty"] and addr("null", "US")["a_empty"]


def test_postcode_vs_house_number():
    a = addr("Suite 200, 1515 Broadway, New York, NY 10036", "US")
    assert a["a_postcode"] == "10036" and a["a_hn"] == "1515" and a["a_unit"] == "200" and a["a_state"] == "NY"
    b = addr("59000 Lille, 12 rue X", "France")
    assert b["a_postcode"] == "59000" and b["a_city"] == "lille" and b["a_hn"] == "12"


def test_france():
    a = addr("00354 R. DE LANNOY, ROUBAIX, Hauts-de-France", "France")
    assert a["a_hn_raw"] == "00354" and a["a_hn"] == "354" and a["a_street"] == "rue lannoy" and a["a_state"] == "HDF"
    b = addr("18 RUE JEN ZAY, Dunkerque, Nord", "France")
    assert b["a_state"] == "HDF" and b["a_city"] == "dunkerque"   # department -> region
    c = addr("5 bis Rue Pierre Dignac, La Teste-de-Buch", "France")
    assert c["a_hn"] == "5" and c["a_hn_suffix"] == "bis" and c["a_city"] == "la teste de buch"


def test_india():
    a = addr("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India")
    assert a["a_state"] == "DL" and a["a_hn"] == "570" and a["a_hn_suffix"] == "13"
    b = addr("Near SBI ATM, MG Road, Bangalore 560001, Karnataka", "India")
    assert b["a_landmark"] and b["a_postcode"] == "560001" and b["a_city"] == "bangalore" and b["a_state"] == "KA"
    c = addr("F-25, Asha Complex, Sec-18, Lucknow, UP", "India")
    assert c["a_hn"] == "25" and c["a_state"] == "UP" and c["a_city"] == "lucknow"


def test_unknown_country_still_parses():
    a = addr("12 Hauptstrasse, Berlin, 10115", "Germany")
    assert a["a_hn"] == "12" and a["a_postcode"] == "10115" and not a["a_empty"]


def test_city_lexicon():
    lex = build_city_lexicon(["1 Main St, Euclid, OH", "2 Main St, Euclid, OH", "3 Rd, Lucknow, UP", "4 Rd, Lucknow 226001, UP"],
                             ["US", "US", "India", "India"], TL, min_count=2)
    assert lex == {"US": {"euclid"}, "India": {"lucknow"}}
