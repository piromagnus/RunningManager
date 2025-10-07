from utils.formatting import set_locale, fmt_decimal, fmt_km, fmt_m, fmt_speed_kmh, to_str_storage


def test_fr_formatting():
    set_locale("fr_FR")
    s = fmt_decimal(1234.5)
    # Normalize NBSP (\u00A0) and NNBSP (\u202F) to regular space for assertion
    normalized = s.replace("\u00A0", " ").replace("\u202F", " ")
    assert normalized == "1 234,5"
    assert fmt_km(12.3).endswith("km")
    assert "," in fmt_km(12.3)
    assert fmt_m(350).endswith("m")
    assert fmt_speed_kmh(9.4).endswith("km/h")
    assert to_str_storage(12.3) == "12.3"
