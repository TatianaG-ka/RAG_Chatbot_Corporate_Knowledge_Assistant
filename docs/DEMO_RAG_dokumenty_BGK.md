# Demo RAG — które dokumenty BGK wczytać (z linkami)

> Zestaw publicznych dokumentów BGK do zbudowania „Asystenta Wiedzy BGK" na demo.
> Wszystko publiczne (bgk.pl) — zero ryzyka poufności. Linki zweryfikowane (czerwiec 2026).

## Zasady
- **Tylko dokumenty publiczne** z bgk.pl. Nie używaj niczego poufnego.
- **Mały zbiór** (5–15 plików) → szybciej i trafniej niż 100 dokumentów.
- **Cytuj nazwę pliku źródłowego** w odpowiedzi (to esencja Twojego use case'u).
- Musi być **luka** → żeby pokazać `refuse-on-no-context`.
- **Zweryfikuj odpowiedzi przed rozmową** — zero niespodzianek z halucynacją na żywo.

---

## Zestaw rekomendowany (priorytet od góry)

### 1. ⭐ Pożyczka na cyfryzację (KPO) — najlepszy wybór tematyczny
Wprost AI/cyfryzacja, 2,8 mld zł, narracja „BGK już finansuje cyfryzację".
- Strona produktu: https://www.bgk.pl/produkty/pozyczka-na-cyfryzacje/
- **Zasady naboru wniosków (PDF):** https://www.bgk.pl/files/public/Grafika/KPO/Po%C5%BCyczka_na_cyfryzacj%C4%99/Zasady_naboru_wniosk%C3%B3w_dla_Po%C5%BCyczek_na_cyfryzacj%C4%99.pdf
- Klauzula informacyjna / RODO (PDF): https://www.bgk.pl/files/public/Grafika/KPO/Po%C5%BCyczka_na_cyfryzacj%C4%99/za%C5%82%C4%85cznik_nr_2_-_Klauzula_informacyjna_Banku_Gospodarstwa_Krajowego.pdf

**Fakty do pytań:** min. kwota 5 mln zł · do 100% kosztów netto · JST i uczelnie nieoprocentowane, firmy (de minimis) 0,5% · nabór otwarty do 30.09.2025, zawieszony od 1.10.2025 · umowy do 31.08.2026.

### 2. ⭐ Gwarancja de minimis — najbogatszy w konkretne warunki
Klasyczny program, dużo liczb (kwoty, %, okresy) → świetne pytania testowe.
- Strona produktu: https://www.bgk.pl/produkty/gwarancja-de-minimis/
- **Warunki uzyskania (PDF, od 16.04.2026):** https://www.bgk.pl/files/public/Pliki/Przedsiebiorstwa/Gwarancje_de_minimis/Warunki_uzyskania_gwarancji_de_minimis_od_16-04-2026.pdf
- FAQ — najczęstsze pytania: https://www.bgk.pl/wazna-informacja-dla-klientow/gwarancje-de-minimis-najczestsze-pytania-i-odpowiedzi-o-nowe-zasady/

**Fakty do pytań:** dla MŚP · gwarancja do **60%** kwoty kredytu · do 60 mies. (obrotowy) / 120 mies. (inwestycyjny) · prowizja 0,5%.
⚠️ Uwaga: na slajdzie DEMO makieta mówi „do 80%" — dla **de minimis to 60%** (80% dotyczy Biznesmax/Ekomax). Wyrównaj odpowiedź na żywo do dokumentu.

### 3. Gwarancje FENG: Biznesmax Plus / Ekomax — drugi obszar dla kontrastu
- Biznesmax Plus: https://www.bgk.pl/produkty/gwarancja-biznesmax-plus/
- Ekomax: https://www.bgk.pl/produkty/gwarancja-ekomax/
- Warunki FENG (PDF): https://www.bgk.pl/files/public/Pliki/Przedsiebiorstwa/Gwarancje_POIR/Gwarancja_Biznesmax_Plus/BGK_Warunki_uzyskania_gwarancji_z_FG_FENG_i_doplaty_09_2024.pdf
- Przewodnik po kryteriach FENG (PDF): https://www.bgk.pl/files/public/Pliki/Przedsiebiorstwa/Gwarancja_Ekomax/Przewodnik_po_kryteriach_Gwarancje_FENG_v_02_2024.pdf

**Fakty:** do **80%** kwoty kredytu, maks. 2,5 mln EUR · Biznesmax dla projektów innowacyjnych/cyfrowych, Ekomax dla efektywności energetycznej (audyt energetyczny min. 30% oszczędności).

### 4. Strategia BGK 2030 (masz już PDF lokalnie)
Do pytań „co BGK robi / dokąd zmierza". Plik: `docs/Prezentacja_strategii_BGK_2025-2030*.pdf`.

---

## Scenariusz demo — 5 pytań (przećwicz wcześniej)

| # | Pytanie | Oczekiwana odpowiedź |
|---|---|---|
| 1 | „Jaka jest minimalna kwota Pożyczki na cyfryzację i kto może wnioskować?" | 5 mln zł; firmy, JST, uczelnie — **z cytatem** do Zasad naboru |
| 2 | „Do jakiej części kredytu sięga gwarancja de minimis?" | do 60% — **z cytatem** do Warunków |
| 3 | „Co finansuje gwarancja Biznesmax, a co gwarancja Ekomax?" | innowacje/cyfryzacja vs efektywność energetyczna — **z cytatem** |
| 4 | „Jaki jest okres gwarancji dla kredytu inwestycyjnego de minimis?" | do 120 mies. — **z cytatem** |
| 5 | **PUŁAPKA (refuse):** „Czy gwarancja de minimis obejmuje kredyt hipoteczny dla osoby fizycznej?" | *„Nie wiem — brak podstawy w dokumentach"* (program dotyczy MŚP, nie konsumentów) |

Pytanie 5 to najważniejszy moment — pokazuje, że system **nie zmyśla**. To różnica między narzędziem dla banku a chatbotem z internetu.

---

## Po demie — jedno zdanie
> *„To działa na publicznych dokumentach BGK — czyli to nie jest 'RAG w ogóle', tylko Asystent Wiedzy BGK z mojego flagowego use case'u. Na danych wewnętrznych w tenancie działa tak samo, tylko bezpieczniej."*
